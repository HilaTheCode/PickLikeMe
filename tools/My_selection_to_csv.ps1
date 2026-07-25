# ============================
# Configure your folders
# ============================

$selectedRoot = "C:\Users\hila7\Pictures\_Selected"
$rejectedRoot = "C:\Users\hila7\Pictures\_Rejected"

$outputCsv = "C:\my_selection.csv"

# ============================
# Collect Selected
# ============================

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

# ============================
# Collect Rejected
# ============================

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

# ============================
# Merge and export
# ============================

$selected + $rejected |
Export-Csv $outputCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Done!"
Write-Host "CSV saved to:"
Write-Host $outputCsv