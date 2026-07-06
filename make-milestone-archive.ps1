[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Label = "milestone",

    [string]$OutputDirectory = ".bak\milestones",

    [switch]$TrackedOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath(
    (Split-Path -Parent $MyInvocation.MyCommand.Path)
).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$ProjectName = Split-Path -Leaf $ProjectRoot

function Invoke-GitCapture {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $Result = & git -C $ProjectRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git -C `"$ProjectRoot`" $($Arguments -join ' ')"
    }
    return $Result
}

function Get-GitTopLevel {
    $TopLevel = & git -C $ProjectRoot rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($TopLevel)) {
        return $null
    }

    return [System.IO.Path]::GetFullPath($TopLevel.Trim()).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-SamePath {
    param([string]$Left, [string]$Right)
    return [string]::Equals(
        $Left,
        $Right,
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not available on PATH. Install Git for Windows before running this script."
}

$GitTopLevel = Get-GitTopLevel
if ($null -eq $GitTopLevel) {
    throw "No Git repository was found. Run setup-local-git.ps1 first."
}
if (-not (Test-SamePath $GitTopLevel $ProjectRoot)) {
    throw "Refusing to archive from parent repository '$GitTopLevel'. Run the corrected setup-local-git.ps1 to create a repository rooted at '$ProjectRoot'."
}

$SafeLabel = $Label.Trim()
if ([string]::IsNullOrWhiteSpace($SafeLabel)) {
    $SafeLabel = "milestone"
}
$SafeLabel = $SafeLabel -replace '[^A-Za-z0-9._-]+', '-'
$SafeLabel = $SafeLabel.Trim('-')
if ([string]::IsNullOrWhiteSpace($SafeLabel)) {
    $SafeLabel = "milestone"
}

if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $ResolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    $ResolvedOutputDirectory = [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $OutputDirectory)
    )
}
New-Item -ItemType Directory -Force -Path $ResolvedOutputDirectory | Out-Null

if ($TrackedOnly) {
    $Files = @(Invoke-GitCapture ls-files)
} else {
    # Includes tracked files plus untracked files that are not excluded by .gitignore.
    $Files = @(Invoke-GitCapture ls-files --cached --others --exclude-standard)
}

$Files = @(
    $Files |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_) -and
            $_ -notmatch '(^|/)\.git(/|$)' -and
            $_ -notmatch '(^|/)\.bak(/|$)'
        } |
        Sort-Object -Unique
)

if ($Files.Count -eq 0) {
    throw "Git found no eligible project files to archive."
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ArchiveName = "$ProjectName-$SafeLabel-$Timestamp.zip"
$ArchivePath = Join-Path $ResolvedOutputDirectory $ArchiveName

$StagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "jlc-flux2-controlnet-archive-" + [Guid]::NewGuid().ToString("N")
)
$StagingProject = Join-Path $StagingRoot $ProjectName
New-Item -ItemType Directory -Force -Path $StagingProject | Out-Null

try {
    $ArchivedFiles = New-Object System.Collections.Generic.List[string]

    foreach ($RelativePath in $Files) {
        # Git emits forward-slash relative paths even on Windows.
        $NativeRelativePath = $RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar
        $SourcePath = [System.IO.Path]::GetFullPath(
            (Join-Path $ProjectRoot $NativeRelativePath)
        )

        # Defense in depth: no selected path may escape the project root.
        $ProjectPrefix = $ProjectRoot + [System.IO.Path]::DirectorySeparatorChar
        if (-not $SourcePath.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing path outside project root: $RelativePath"
        }

        if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
            Write-Warning "Skipping missing file listed by Git: $RelativePath"
            continue
        }

        $DestinationPath = Join-Path $StagingProject $NativeRelativePath
        $DestinationParent = Split-Path -Parent $DestinationPath
        if ($DestinationParent) {
            New-Item -ItemType Directory -Force -Path $DestinationParent | Out-Null
        }

        Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
        $ArchivedFiles.Add($RelativePath)
    }

    if ($ArchivedFiles.Count -eq 0) {
        throw "No eligible files were copied into the milestone archive."
    }

    $Commit = (& git -C $ProjectRoot rev-parse --verify HEAD 2>$null)
    if ($LASTEXITCODE -ne 0) {
        $Commit = "<no commits yet>"
    }

    $Branch = (& git -C $ProjectRoot branch --show-current 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Branch)) {
        $Branch = "<unborn or detached>"
    }

    $Status = @(& git -C $ProjectRoot status --short)
    if ($LASTEXITCODE -ne 0) {
        $Status = @("<unable to read status>")
    } elseif ($Status.Count -eq 0) {
        $Status = @("<clean working tree>")
    }

    $ManifestLines = @(
        "JLC Flux2 ControlNet Milestone Archive",
        "======================================",
        "",
        "Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
        "Label: $Label",
        "Project: $ProjectName",
        "Repository root: $GitTopLevel",
        "Branch: $Branch",
        "Commit: $Commit",
        "Archive mode: $(if ($TrackedOnly) { 'tracked files only' } else { 'tracked plus non-ignored untracked files' })",
        "Archived files: $($ArchivedFiles.Count)",
        "",
        "Git status at archive time:",
        $Status,
        "",
        "Archived files:",
        $ArchivedFiles
    )

    $ManifestPath = Join-Path $StagingProject "MILESTONE_MANIFEST.txt"
    $ManifestLines | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }

    Compress-Archive -Path $StagingProject -DestinationPath $ArchivePath -CompressionLevel Optimal

    Write-Host "Milestone archive created:" -ForegroundColor Green
    Write-Host "  $ArchivePath"
    Write-Host "Files archived: $($ArchivedFiles.Count)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The archive contains current working copies, not only the last commit." -ForegroundColor Yellow
} finally {
    if (Test-Path -LiteralPath $StagingRoot) {
        Remove-Item -LiteralPath $StagingRoot -Recurse -Force
    }
}
