Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'

function Resolve-Python {
    foreach ($cand in @('python', 'py')) {
        try {
            $v = & $cand --version 2>$null
            if ($LASTEXITCODE -eq 0) { return $cand }
        } catch {}
    }
    throw 'Python interpreter not found. Please install Python 3 and ensure it is on PATH.'
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $repoRoot '..')
$script = Join-Path $repoRoot 'my_scripts/align_my_documents.py'

$py = Resolve-Python
& $py $script @Args
exit $LASTEXITCODE

