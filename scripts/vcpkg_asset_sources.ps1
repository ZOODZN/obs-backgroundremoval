# SPDX-FileCopyrightText: 2026 Kaito Udagawa <umireon@kaito.tokyo>
#
# SPDX-License-Identifier: Apache-2.0

# file: scripts/vcpkg_asset_sources.ps1
# description: Helper script to download vcpkg assets.
# author: Kaito Udagawa <umireon@kaito.tokyo>
# date: 2026-06-21

param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [string]$Sha512,
    [Parameter(Mandatory = $true)]
    [string]$Dst
)

Set-StrictMode -Version Latest; $ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $true

$curlOpts = @(
    '-fL',
    '--retry', '5',
    '--retry-all-errors'
)

if ($Url.StartsWith('https://github.com/') -and $env:GITHUB_TOKEN) {
    $curlOpts += @(
        '-H', "Authorization: Bearer $env:GITHUB_TOKEN"
    )
}

curl $curlOpts $Url --output $Dst
