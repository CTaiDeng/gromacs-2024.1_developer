param(
    [string]$complexPdb = "complex.pdb",
    [string]$ligandResidue = "LIG"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $scriptDir "..")
$outRoot = Join-Path $root "out"
if (!(Test-Path $outRoot)) {
    New-Item -ItemType Directory -Path $outRoot | Out-Null
}
$jobDir = Join-Path $outRoot ("gmx_split_{0:yyyyMMdd_HHmmss}" -f (Get-Date))
New-Item -ItemType Directory -Path $jobDir | Out-Null
Write-Host "Working directory: $jobDir"

$complexGro = Join-Path $jobDir "complex.gro"
$ligandNdx = Join-Path $jobDir "ligand.ndx"
$ligandPdb = Join-Path $jobDir "ligand.pdb"
$receptorNdx = Join-Path $jobDir "receptor.ndx"
$receptorPdb = Join-Path $jobDir "receptor.pdb"

# Convert PDB to GRO for consistent processing
& gmx editconf -quiet -f $complexPdb -o $complexGro | Out-Null

# Select ligand atoms
$selLigand = "resname $ligandResidue"
& gmx select -quiet -f $complexGro -s $complexGro -on $ligandNdx -select $selLigand | Out-Null

# Dump ligand coordinates (group 0 from the generated index)
cmd /c "echo 0 | gmx trjconv -quiet -f `"$complexGro`" -s `"$complexGro`" -o `"$ligandPdb`" -n `"$ligandNdx`" -dump 0" | Out-Null

# Select receptor (everything except ligand)
$selReceptor = "not resname $ligandResidue"
& gmx select -quiet -f $complexGro -s $complexGro -on $receptorNdx -select $selReceptor | Out-Null
cmd /c "echo 0 | gmx trjconv -quiet -f `"$complexGro`" -s `"$complexGro`" -o `"$receptorPdb`" -n `"$receptorNdx`" -dump 0" | Out-Null

Write-Host "Ligand written to $ligandPdb"
Write-Host "Receptor written to $receptorPdb"
