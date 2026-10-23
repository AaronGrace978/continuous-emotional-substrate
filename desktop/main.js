const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const serverScript = path.join(repoRoot, "chat_server.py");
const indexFile = path.join(__dirname, "renderer", "index.html");
const venvPython = path.join(
  process.env.LOCALAPPDATA || "",
  "ChronoSoul",
  "venv",
  "Scripts",
  "python.exe"
);

let win = null;
let serverProc = null;
let serverUrl = "";
let bootPromise = null;
const pending = [];

function send(payload) {
  if (win && !win.isDestroyed() && !win.webContents.isLoading()) {
    win.webContents.send("boot", payload);
  } else {
    pending.push(payload);
  }
}

function flush() {
  if (!win || win.isDestroyed() || win.webContents.isLoading()) return;
  while (pending.length) {
    win.webContents.send("boot", pending.shift());
  }
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd: repoRoot,
      windowsHide: true,
      env: process.env,
    });
    let buf = "";
    const onData = (chunk) => {
      const text = chunk.toString();
      buf += text;
      const line = text
        .replace(/\r/g, "\n")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
        .pop();
      if (line) send({ phase: "packages", detail: line.slice(0, 220) });
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve(buf);
      else {
        reject(
          new Error(
            `${path.basename(cmd)} failed (${code})\n${buf.slice(-1200)}`
          )
        );
      }
    });
  });
}

function probe(py, code) {
  try {
    return execFileSync(py, ["-c", code], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 180000,
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  } catch {
    return "";
  }
}

function hasGpu() {
  try {
    execFileSync("nvidia-smi", ["-L"], {
      stdio: "ignore",
      windowsHide: true,
      timeout: 15000,
    });
    return true;
  } catch {
    return false;
  }
}

async function installTorch(py) {
  const gpu = hasGpu();
  const attempts = gpu
    ? [
        ["torch==2.14.1+cu130", "https://download.pytorch.org/whl/cu130"],
        ["torch", "https://download.pytorch.org/whl/cu128"],
        ["torch", "https://download.pytorch.org/whl/cpu"],
      ]
    : [["torch", "https://download.pytorch.org/whl/cpu"]];
  let lastError = null;
  for (const [spec, index] of attempts) {
    send({
      phase: "packages",
      detail: gpu
        ? "Installing PyTorch for the GPU. The first launch is a large download."
        : "Installing the CPU build of PyTorch.",
    });
    try {
      await run(py, [
        "-m",
        "pip",
        "install",
        spec,
        "--index-url",
        index,
        "--extra-index-url",
        "https://pypi.org/simple",
      ]);
      return;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("PyTorch could not be installed.");
}

async function ensurePython() {
  send({ phase: "runtime", detail: "Checking the local mind runtime…" });
  if (fs.existsSync(venvPython)) {
    const cuda = probe(
      venvPython,
      "import torch; print('1' if torch.cuda.is_available() else '0')"
    );
    if (cuda === "1") return venvPython;
    const libs = probe(
      venvPython,
      "import torch, transformers, peft; print('ok')"
    );
    if (libs === "ok" && !hasGpu()) return venvPython;
  }

  const venvDir = path.dirname(path.dirname(venvPython));
  if (!fs.existsSync(venvPython)) {
    send({ phase: "packages", detail: "Creating a private Python environment…" });
    await run("python", ["-m", "venv", venvDir]);
  }
  await run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
  await installTorch(venvPython);
  await run(venvPython, [
    "-m",
    "pip",
    "install",
    "-r",
    path.join(__dirname, "requirements.txt"),
  ]);
  if (hasGpu()) {
    try {
      await run(venvPython, ["-m", "pip", "install", "bitsandbytes>=0.45.0"]);
    } catch {
      /* 4-bit quantization is optional; accelerate offload stays available */
    }
  }
  const ok = probe(
    venvPython,
    "import torch, transformers, peft; print('ok')"
  );
  if (ok === "ok") return venvPython;
  const sysOk = probe("python", "import torch, transformers, peft; print('ok')");
  if (sysOk === "ok") return "python";
  throw new Error("PyTorch, transformers, and peft are not importable.");
}

function killServer() {
  if (!serverProc) return;
  const pid = serverProc.pid;
  try {
    serverProc.kill();
  } catch {
    /* already gone */
  }
  if (pid) {
    spawn("taskkill", ["/pid", String(pid), "/t", "/f"], { windowsHide: true });
  }
  serverProc = null;
}

function startServer(py) {
  return new Promise((resolve, reject) => {
    if (serverProc && serverUrl) {
      resolve(serverUrl);
      return;
    }
    const child = spawn(py, [serverScript], {
      cwd: repoRoot,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        HF_HUB_DISABLE_TELEMETRY: "1",
        HF_HUB_DISABLE_SYMLINKS_WARNING: "1",
        TOKENIZERS_PARALLELISM: "false",
      },
    });
    serverProc = child;
    let out = "";
    let err = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(
        new Error(
          "The mind process did not open a port in time.\n" + err.slice(-1500)
        )
      );
    }, 90000);

    const finish = (fn) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };

    child.stdout.on("data", (chunk) => {
      out += chunk.toString();
      const match = out.match(/CHRONOSOUL_READY\s+(\d+)/);
      if (match) {
        finish(() => {
          serverUrl = `http://127.0.0.1:${match[1]}`;
          resolve(serverUrl);
        });
      }
    });
    child.stderr.on("data", (chunk) => {
      err += chunk.toString();
      if (err.length > 8000) err = err.slice(-8000);
      const line = chunk
        .toString()
        .replace(/\r/g, "\n")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
        .pop();
      if (line && !settled) send({ phase: "server", detail: line.slice(0, 220) });
    });
    child.on("error", (error) => finish(() => reject(error)));
    child.on("exit", (code) => {
      serverProc = null;
      finish(() => {
        reject(
          new Error(`Mind process exited (${code}).\n${err.slice(-1500)}`)
        );
      });
    });
  });
}

async function bootInner() {
  const py = await ensurePython();
  send({ phase: "server", detail: "Opening the chamber…" });
  const url = await startServer(py);
  send({ phase: "online", url, detail: "Chamber open" });
  if (win && !win.isDestroyed()) await win.loadURL(url);
}

function boot() {
  if (!bootPromise) {
    bootPromise = bootInner().catch((err) => {
      bootPromise = null;
      send({ phase: "error", detail: String(err && err.message ? err.message : err) });
    });
  }
  return bootPromise;
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 1060,
    minHeight: 700,
    backgroundColor: "#12100e",
    frame: false,
    show: false,
    autoHideMenuBar: true,
    title: "ChronoSoul",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.once("ready-to-show", () => win.show());
  win.webContents.on("did-finish-load", flush);
  win.loadFile(indexFile);
  win.on("closed", () => {
    win = null;
  });
}

ipcMain.on("win", (_event, action) => {
  if (!win || win.isDestroyed()) return;
  if (action === "min") win.minimize();
  else if (action === "max") {
    if (win.isMaximized()) win.unmaximize();
    else win.maximize();
  } else if (action === "close") win.close();
});

ipcMain.handle("retry-boot", () => {
  bootPromise = null;
  return boot();
});

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!win || win.isDestroyed()) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });

  let quitting = false;
  app.on("before-quit", (event) => {
    if (quitting) return;
    quitting = true;
    if (!serverUrl) {
      killServer();
      return;
    }
    event.preventDefault();
    let closed = false;
    const finish = () => {
      if (closed) return;
      closed = true;
      killServer();
      app.quit();
    };
    const req = http.request(serverUrl + "/shutdown", { method: "POST" }, (res) => {
      res.resume();
      finish();
    });
    req.setTimeout(2500, () => req.destroy());
    req.on("error", finish);
    req.end();
  });

  app.whenReady().then(() => {
    app.setAppUserModelId("chronosoul.chamber");
    createWindow();
    boot();
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}
