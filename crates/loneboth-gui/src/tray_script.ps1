[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notifyIcon = New-Object System.Windows.Forms.NotifyIcon
$notifyIcon.Icon = [System.Drawing.SystemIcons]::Shield
$notifyIcon.Text = "loneboth: 生体感覚統御コア"
$notifyIcon.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenuStrip

$statusItem = New-Object System.Windows.Forms.ToolStripMenuItem("ホメオスタシス: 待機中")
$statusItem.Enabled = $false

$toggleUiItem = New-Object System.Windows.Forms.ToolStripMenuItem("生体モニターUI 表示/非表示")
$toggleUiItem.Add_Click({
    [Console]::Out.WriteLine("TOGGLE_UI")
    [Console]::Out.Flush()
})

$exitItem = New-Object System.Windows.Forms.ToolStripMenuItem("終了")
$exitItem.Add_Click({
    [Console]::Out.WriteLine("EXIT")
    [Console]::Out.Flush()
    $notifyIcon.Visible = $false
    $notifyIcon.Dispose()
    [System.Windows.Forms.Application]::Exit()
})

[void]$menu.Items.Add($statusItem)
[void]$menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))
[void]$menu.Items.Add($toggleUiItem)
[void]$menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))
[void]$menu.Items.Add($exitItem)

$notifyIcon.ContextMenuStrip = $menu

$queue = New-Object 'System.Collections.Concurrent.ConcurrentQueue[string]'
[void][System.Threading.Tasks.Task]::Run([Action]{
    while ($true) {
        $line = [Console]::In.ReadLine()
        if ($null -eq $line) {
            $queue.Enqueue("QUIT")
            break
        }
        $queue.Enqueue($line)
    }
})

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 150
$timer.Add_Tick({
    $msg = $null
    while ($queue.TryDequeue([ref]$msg)) {
        if ($msg -eq "QUIT") {
            $notifyIcon.Visible = $false
            $notifyIcon.Dispose()
            [System.Windows.Forms.Application]::Exit()
            return
        }
        if ($msg.StartsWith("STATUS|")) {
            $parts = $msg.Split("|", 3)
            if ($parts.Length -ge 3) {
                $kind = $parts[1]
                $title = $parts[2]
                $notifyIcon.Icon = if ($kind -eq "alert") { [System.Drawing.SystemIcons]::Warning } else { [System.Drawing.SystemIcons]::Application }
                $statusItem.Text = $title
                if ($title.Length -gt 63) { $title = $title.Substring(0, 63) }
                $notifyIcon.Text = $title
            }
        }
    }
})
$timer.Start()

[System.Windows.Forms.Application]::Run()
