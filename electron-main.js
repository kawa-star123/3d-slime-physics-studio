'use strict';

const { app, BrowserWindow, Menu, shell } = require('electron');
const path = require('path');

const isMac = process.platform === 'darwin';

function buildMenu() {
  const template = [
    ...(isMac ? [{ role: 'appMenu' }] : []),
    {
      label: 'ファイル',
      submenu: [isMac ? { role: 'close' } : { role: 'quit', label: '終了' }],
    },
    {
      label: '表示',
      submenu: [
        { role: 'reload', label: '再読み込み' },
        { role: 'forceReload', label: '強制再読み込み' },
        { role: 'toggleDevTools', label: '開発者ツール' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'ズームをリセット' },
        { role: 'zoomIn', label: '拡大' },
        { role: 'zoomOut', label: '縮小' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'フルスクリーン' },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() { // メインウィンドウの作成及び設定
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    title: 'うごけスライム',
    con: path.join(__dirname, 'icon.ico'), //アイコン適用
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#F8FAFC', // 起動時のちらつきを止める
    title: 'Multi-View 3D Soft-Body Physics Studio', //ウィンドウ上部に表示されるタイトル
    autoHideMenuBar: true, // 普段はメニューバーを隠す
    show: false, //準備ができるまでウィンドウを隠す
    webPreferences: { //以下セキュリティ設定
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  win.once('ready-to-show', () => win.show());
  win.loadFile(path.join(__dirname, 'index.html'));

  // 外部リンクは既定ブラウザで開き、アプリ内ナビゲーションは許可しない
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('file://')) event.preventDefault();
  });
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (!isMac) app.quit();
});