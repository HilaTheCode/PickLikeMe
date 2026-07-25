Add-Type -AssemblyName System.Windows.Forms

function Select-Folder($description)
{
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $description
    $dialog.ShowNewFolderButton = $false

    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)
    {
        return $dialog.SelectedPath
    }

    Write-Host "Operation cancelled."
    exit
}

function Select-OutputFile()
{
    $dialog = New-Object System.Windows.Forms.SaveFileDialog
    $dialog.Filter = "CSV files (*.csv)|*.csv"
    $dialog.DefaultExt = "csv"
    $dialog.FileName = "labels.csv"

    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)
    {
        return $dialog.FileName
    }

    Write-Host "Operation cancelled."
    exit
}

Write-Host ""
Write-Host "======================================="
Write-Host " PickLikeMe Label Export"
Write-Host "======================================="
Write-Host ""

$selectedRoot = Select-Folder "Select the SELECTED folder"

$rejectedRoot = Select-Folder "Select the REJECTED folder"

$outputCsv = Select-OutputFile

Write-Host ""
Write-Host "Selected folder : $selectedRoot"
Write-Host "Rejected folder : $rejectedRoot"
Write-Host "Output CSV      : $outputCsv"
Write-Host ""

$selected =
Get-ChildItem -LiteralPath $selectedRoot -Recurse -File |
ForEach-Object {

    [PSCustomObject]@{
        filename      = $_.Name
        relative_path = $_.FullName.Substring($selectedRoot.Length + 1)
        label         = "selected"
        full_path     = $_.FullName
    }
}

$rejected =
Get-ChildItem -LiteralPath $rejectedRoot -Recurse -File |
ForEach-Object {

    [PSCustomObject]@{
        filename      = $_.Name
        relative_path = $_.FullName.Substring($rejectedRoot.Length + 1)
        label         = "rejected"
        full_path     = $_.FullName
    }
}

$selected + $rejected |
Sort-Object full_path |
Export-Csv -Path $outputCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "======================================="
Write-Host "Done!"
Write-Host ""
Write-Host "Exported $($selected.Count) selected images"
Write-Host "Exported $($rejected.Count) rejected images"
Write-Host ""
Write-Host "CSV saved to:"
Write-Host $outputCsv
Write-Host "======================================="

Read-Host "Press ENTER to exit"