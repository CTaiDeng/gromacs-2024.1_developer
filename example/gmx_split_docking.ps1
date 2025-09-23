param(
    [string]$complexPdb = "res/hiv.pdb",
    [string]$ligandResidue = "CSO"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $scriptDir "..")
$gmxExe = Join-Path $root "cmake-build-release-visual-studio-2022/bin/gmx.exe"
if (!(Test-Path $gmxExe)) {
    throw "Cannot find GROMACS executable: $gmxExe"
}

$complexCandidate = if ([System.IO.Path]::IsPathRooted($complexPdb)) {
    $complexPdb
} else {
    Join-Path $root $complexPdb
}

try {
    $complexPdbPath = (Resolve-Path $complexCandidate).Path
} catch {
    throw "Input coordinate file not found or inaccessible: $complexCandidate"
}

function Invoke-GmxCommand {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$InputText
    )

    if ($PSBoundParameters.ContainsKey('InputText')) {
        $InputText | & $script:gmxExe @Arguments | Out-Null
    } else {
        & $script:gmxExe @Arguments | Out-Null
    }

    if ($LASTEXITCODE -ne 0) {
        throw ("gmx {0} failed with exit code {1}" -f $Arguments[0], $LASTEXITCODE)
    }
}

$outRoot = Join-Path $root "out"
if (!(Test-Path $outRoot)) {
    New-Item -ItemType Directory -Path $outRoot | Out-Null
}
$jobDir = Join-Path $outRoot ("gmx_split_{0:yyyyMMdd_HHmmss}" -f (Get-Date))
New-Item -ItemType Directory -Path $jobDir | Out-Null
Write-Host "Working directory: $jobDir"

$complexCopy = Join-Path $jobDir (Split-Path $complexPdbPath -Leaf)
Copy-Item -Path $complexPdbPath -Destination $complexCopy -Force
$complexInput = $complexCopy

$complexGro  = Join-Path $jobDir "complex.gro"
$ligandNdx   = Join-Path $jobDir "ligand.ndx"
$ligandPdb   = Join-Path $jobDir "ligand.pdb"
$receptorNdx = Join-Path $jobDir "receptor.ndx"
$receptorPdb = Join-Path $jobDir "receptor.pdb"

# Convert PDB to GRO for consistent processing
Invoke-GmxCommand -Arguments @('editconf','-quiet','-f',$complexInput,'-o',$complexGro)

# Select ligand atoms
$selLigand = "resname $ligandResidue"
Invoke-GmxCommand -Arguments @('select','-quiet','-f',$complexGro,'-s',$complexGro,'-on',$ligandNdx,'-select',$selLigand)

# Dump ligand coordinates (group 0 from the generated index)
Invoke-GmxCommand -Arguments @('trjconv','-quiet','-f',$complexGro,'-s',$complexGro,'-o',$ligandPdb,'-n',$ligandNdx,'-dump','0') -InputText "0`n"

# Select receptor (everything except ligand)
$selReceptor = "not resname $ligandResidue"
Invoke-GmxCommand -Arguments @('select','-quiet','-f',$complexGro,'-s',$complexGro,'-on',$receptorNdx,'-select',$selReceptor)
Invoke-GmxCommand -Arguments @('trjconv','-quiet','-f',$complexGro,'-s',$complexGro,'-o',$receptorPdb,'-n',$receptorNdx,'-dump','0') -InputText "0`n"

Write-Host "Copied input structure to $complexCopy"
Write-Host "Ligand written to $ligandPdb"
Write-Host "Receptor written to $receptorPdb"
