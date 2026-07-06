<#
JLC Flux2 ControlNet Map Generator — gitignore-aware text map

Run this from the root of the JLC Flux2 ControlNet repository, or pass -RootPath.

Examples:
  pwsh .\jlc-structure-mapper.ps1
  pwsh .\jlc-structure-mapper.ps1 -RootPath "C:\Users\josel\Stable_Diffusion\ComfyUI_V89\custom_nodes\JLC-Flux2-ControlNet"
  pwsh .\jlc-structure-mapper.ps1 -MaxDepth 4
#>

[CmdletBinding()]
param(
    [string]$RootPath = (Get-Location).Path,
    [string]$OutputFile = "jlc-flux2-controlnet-structure.txt",
    [int]$MaxDepth = 0,
    [switch]$IncludeHidden,
    [switch]$ShowExcluded
)

$ErrorActionPreference = "Stop"
$PathTrimChars = [char[]]@('\', '/')
$SlashTrimChars = [char[]]@('/')

# PowerShell version note. Script remains plain-text and works without emoji.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Host "WARNING: PowerShell $($PSVersionTable.PSVersion) detected. Script should still work, but PowerShell 7+ is preferred."
}
else {
    Write-Host "PowerShell $($PSVersionTable.PSVersion) detected."
}

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        return [System.IO.Path]::GetFullPath($Path)
    }
    catch {
        throw "Could not resolve path: $Path"
    }
}

$RootPath = Resolve-FullPath $RootPath

if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
    throw "RootPath does not exist or is not a folder: $RootPath"
}

if ([System.IO.Path]::IsPathRooted($OutputFile)) {
    $OutputPath = Resolve-FullPath $OutputFile
}
else {
    $OutputPath = Resolve-FullPath (Join-Path $RootPath $OutputFile)
}

$OutputFolder = Split-Path -Path $OutputPath -Parent
if (-not (Test-Path -LiteralPath $OutputFolder -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
}

$GitIgnorePath = Join-Path $RootPath ".gitignore"

# Things that should almost never be shown in a repository structure map.
# Most generated/dev artifacts should come from .gitignore; this is only the hard safety net.
$AlwaysExcludeNames = @(
    ".git",
    ".svn",
    ".hg",
    "project-structure.txt",
    "project-structure.svg",
    "project-structure.html",
    "jlc-flux2-controlnet-structure.txt"
)

function Get-RelativeUnixPath {
    param(
        [Parameter(Mandatory = $true)][string]$FullName,
        [Parameter(Mandatory = $true)][string]$BasePath
    )

    $full = Resolve-FullPath $FullName
    $base = Resolve-FullPath $BasePath
    $base = $base.TrimEnd($PathTrimChars)

    $comparison = [System.StringComparison]::OrdinalIgnoreCase

    if ($full.Equals($base, $comparison)) {
        return ""
    }

    if ($full.StartsWith($base, $comparison)) {
        return ($full.Substring($base.Length).TrimStart($PathTrimChars) -replace "\\", "/")
    }

    return ($full -replace "\\", "/")
}

function Read-GitIgnorePatterns {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return @()
    }

    $patterns = @()

    foreach ($rawLine in (Get-Content -LiteralPath $Path)) {
        $line = $rawLine.Trim()

        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        if ($line.StartsWith("#")) {
            continue
        }

        # Keep negation lines so the later matcher can honor simple ! exceptions.
        $patterns += $line
    }

    return $patterns
}

$GitIgnorePatterns = @(Read-GitIgnorePatterns -Path $GitIgnorePath)

function Test-IgnorePatternMatch {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$LeafName,
        [Parameter(Mandatory = $true)][bool]$IsDirectory
    )

    $patternText = $Pattern.Trim()

    if ([string]::IsNullOrWhiteSpace($patternText)) {
        return $false
    }

    if ($patternText.StartsWith("!")) {
        $patternText = $patternText.Substring(1).Trim()
    }

    if ([string]::IsNullOrWhiteSpace($patternText)) {
        return $false
    }

    $directoryOnly = $patternText.EndsWith("/")
    $rooted = $patternText.StartsWith("/")

    # This mapper intentionally implements the common .gitignore cases used by this repo:
    #   folder/
    #   *.extension
    #   literal-file-name
    #   path/to/folder/
    # It is not intended to be a complete gitignore engine.
    $patternText = $patternText.TrimStart($SlashTrimChars).TrimEnd($SlashTrimChars)

    if ([string]::IsNullOrWhiteSpace($patternText)) {
        return $false
    }

    if ($directoryOnly -and -not $IsDirectory) {
        return $false
    }

    $hasSlash = $patternText.Contains("/")

    if (-not $hasSlash) {
        return ($LeafName -like $patternText)
    }

    if ($rooted) {
        return (
            ($RelativePath -like $patternText) -or
            ($RelativePath -like "$patternText/*")
        )
    }

    return (
        ($RelativePath -like $patternText) -or
        ($RelativePath -like "*/$patternText") -or
        ($RelativePath -like "$patternText/*") -or
        ($RelativePath -like "*/$patternText/*")
    )
}

function Test-GitIgnored {
    param([Parameter(Mandatory = $true)][System.IO.FileSystemInfo]$Item)

    $relativePath = Get-RelativeUnixPath -FullName $Item.FullName -BasePath $RootPath
    $leafName = $Item.Name
    $isDirectory = [bool]$Item.PSIsContainer

    # Gitignore uses "last matching pattern wins"; support simple negation with !.
    $ignored = $false

    foreach ($pattern in $GitIgnorePatterns) {
        $isNegation = $pattern.Trim().StartsWith("!")

        if (Test-IgnorePatternMatch -Pattern $pattern -RelativePath $relativePath -LeafName $leafName -IsDirectory $isDirectory) {
            $ignored = -not $isNegation
        }
    }

    return $ignored
}

function Test-Excluded {
    param([Parameter(Mandatory = $true)][System.IO.FileSystemInfo]$Item)

    $itemPath = Resolve-FullPath $Item.FullName
    if ($itemPath.Equals($OutputPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    if ($AlwaysExcludeNames -contains $Item.Name) {
        return $true
    }

    if (-not $IncludeHidden) {
        $isHidden = (($Item.Attributes -band [System.IO.FileAttributes]::Hidden) -ne 0)

        # Keep .gitignore visible because it explains the map.
        if ($isHidden -and $Item.Name -ne ".gitignore") {
            return $true
        }
    }

    if (Test-GitIgnored -Item $Item) {
        return $true
    }

    return $false
}

$script:FolderCount = 0
$script:FileCount = 0
$script:ExcludedCount = 0

function Write-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Depth = 0
    )

    $folderName = Split-Path -Path $Path -Leaf
    if ([string]::IsNullOrWhiteSpace($folderName)) {
        $folderName = $Path
    }

    $indent = " " * ($Depth * 2)
    Add-Content -LiteralPath $OutputPath -Value ("$indent$folderName/") -Encoding UTF8
    $script:FolderCount++

    $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue)

    $visibleChildren = @()

    foreach ($child in $children) {
        if (Test-Excluded -Item $child) {
            $script:ExcludedCount++

            if ($ShowExcluded) {
                $childIndent = " " * (($Depth + 1) * 2)
                Add-Content -LiteralPath $OutputPath -Value ("$childIndent[excluded] $($child.Name)") -Encoding UTF8
            }

            continue
        }

        $visibleChildren += $child
    }

    $visibleChildren = @(
        $visibleChildren |
            Sort-Object -Property @{ Expression = { if ($_.PSIsContainer) { 0 } else { 1 } } }, @{ Expression = { $_.Name } }
    )

    if (($MaxDepth -gt 0) -and ($Depth -ge $MaxDepth)) {
        if ($visibleChildren.Count -gt 0) {
            Add-Content -LiteralPath $OutputPath -Value ("$indent  ...") -Encoding UTF8
        }

        return
    }

    foreach ($child in $visibleChildren) {
        if ($child.PSIsContainer) {
            Write-Tree -Path $child.FullName -Depth ($Depth + 1)
        }
        else {
            Add-Content -LiteralPath $OutputPath -Value ("$indent  $($child.Name)") -Encoding UTF8
            $script:FileCount++
        }
    }
}

# Start fresh.
if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

$header = @(
    "JLC Flux2 ControlNet Project Structure",
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "Root: $RootPath",
    "Output: $OutputPath",
    "Gitignore: $(if (Test-Path -LiteralPath $GitIgnorePath -PathType Leaf) { $GitIgnorePath } else { 'not found' })",
    "MaxDepth: $(if ($MaxDepth -gt 0) { $MaxDepth } else { 'unlimited' })",
    ""
)

Set-Content -LiteralPath $OutputPath -Value $header -Encoding UTF8

Write-Tree -Path $RootPath -Depth 0

$footer = @(
    "",
    "Summary:",
    "  Folders shown: $script:FolderCount",
    "  Files shown:   $script:FileCount",
    "  Items skipped: $script:ExcludedCount",
    "  Gitignore patterns loaded: $($GitIgnorePatterns.Count)"
)

Add-Content -LiteralPath $OutputPath -Value $footer -Encoding UTF8

Write-Host "Generated structure map:"
Write-Host "  $OutputPath"
Write-Host "Folders shown: $script:FolderCount"
Write-Host "Files shown:   $script:FileCount"
Write-Host "Items skipped: $script:ExcludedCount"
