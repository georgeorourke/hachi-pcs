use std::io::{Write, stdout};

/// Display progress bar.
pub fn progress_bar(title: &str, cur: usize, total: usize) {
    let pc = ((cur + 1) * 100) / total;
    print!("\r{}: ", title);
    print!("[{:■<1$}", "", pc);
    print!("{:□<1$}]", "", 100-pc);

    if pc == 100 {
        print!("\n");
    }

    let _ = stdout().flush();
}

/// Display a ticked item.
pub fn tick_item(title: &str) {
    println!("{}: ✓", title);
}