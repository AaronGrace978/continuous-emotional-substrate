const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("chronosoul", {
  onBoot: (cb) => {
    ipcRenderer.on("boot", (_event, data) => cb(data));
  },
  retry: () => ipcRenderer.invoke("retry-boot"),
  win: {
    min: () => ipcRenderer.send("win", "min"),
    max: () => ipcRenderer.send("win", "max"),
    close: () => ipcRenderer.send("win", "close"),
  },
});
