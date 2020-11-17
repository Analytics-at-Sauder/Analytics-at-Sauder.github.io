Get-ChildItem 'C:\Users\Charlie\Documents\UBC\Internship\Analytics-at-Sauder.github.io\*.html' -Recurse | 
ForEach {
    (Get-Content $_ | 
    ForEach { $_ -replace '<a', '<a target="_blank"' }) |
    Set-Content $_
    }