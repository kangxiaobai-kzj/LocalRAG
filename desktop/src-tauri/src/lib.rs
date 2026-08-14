// ============================================
// 项目：LocalRAG · 可定制本地 RAG 智能体（桌面壳）
// 开发者：kangxiaobai-kzj
// 开发时间：2026-08-14
// ============================================

// LocalRAG 桌面壳：启动时拉起本地 Streamlit 服务（start_agent.bat），
// 就绪后将 WebView 导航到 http://127.0.0.1:8501；关闭窗口即终止服务进程树。
use std::path::PathBuf;
use std::process::Child;
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

/// 后台服务子进程句柄
struct ServerProc(Mutex<Option<Child>>);

/// 在可执行文件所在目录向上搜索，找到包含 start_agent.bat 的项目根目录。
/// 覆盖三种部署形态：dev（target/debug 向上 3 层）、
/// 与项目并排放置（exe 同目录）、安装到项目内（exe 同目录）。
fn find_project_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent()?.to_path_buf();
    for _ in 0..5 {
        if dir.join("start_agent.bat").exists() {
            return Some(dir);
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    None
}

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// 以隐藏窗口方式启动服务脚本（无控制台弹出）
fn spawn_server(root: &PathBuf) -> std::io::Result<Child> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        return std::process::Command::new("cmd")
            .args(["/C", "start_agent.bat"])
            .current_dir(root)
            .env("LOCALRAG_DESKTOP", "1")
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
    }
    #[cfg(not(windows))]
    {
        let _ = root;
        unreachable!("桌面壳仅支持 Windows")
    }
}

/// 终止进程树（taskkill /T /F）
fn kill_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
    }
}

/// 轮询本地服务健康检查，就绪后把主窗口导航到应用地址。
/// navigate 是同步方法，用独立线程轮询即可，无需 async runtime。
fn wait_server_and_navigate(app: &tauri::App) {
    let window = app.get_webview_window("main");
    let url = "http://127.0.0.1:8501";

    std::thread::spawn(move || {
        let mut attempts = 0u32;
        loop {
            let ready = std::net::TcpStream::connect("127.0.0.1:8501").is_ok();
            if ready {
                if let Some(w) = window.as_ref() {
                    if let Ok(u) = url.parse::<tauri::Url>() {
                        let _ = w.navigate(u);
                    }
                }
                break;
            }
            attempts += 1;
            // 最多等待约 90 秒（模型缺失/环境初始化可能较慢）
            if attempts >= 60 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1500));
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // WebView2 数据目录放到可执行文件旁，避免系统用户目录不可写
            // （如中文用户名或受控环境导致无法写入 %LOCALAPPDATA%）
            let data_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .map(|d| d.join("WebView2Data"))
                .unwrap_or_else(|| std::env::temp_dir().join("localrag-webview2"));
            std::fs::create_dir_all(&data_dir).ok();

            let _window = WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::App("index.html".into()),
            )
            .title("LocalRAG · 可定制本地 RAG 智能体")
            .inner_size(1280.0, 860.0)
            .min_inner_size(1024.0, 700.0)
            .center()
            .resizable(true)
            .data_directory(data_dir)
            .build()?;

            let root = find_project_root();
            if let Some(root) = root {
                if let Ok(child) = spawn_server(&root) {
                    app.manage(ServerProc(Mutex::new(Some(child))));
                }
            }
            wait_server_and_navigate(app);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            RunEvent::Exit => {
                // 应用退出时终止后台服务，避免残留进程占用资源
                if let Some(state) = app.try_state::<ServerProc>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.take() {
                            kill_process_tree(child.id());
                        }
                    }
                }
            }
            _ => {}
        });
}
