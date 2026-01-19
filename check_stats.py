import requests
import json

repo = "Dedsec-b/v6rge-releases-"
url = f"https://api.github.com/repos/{repo}/releases"
headers = {"Accept": "application/vnd.github+json"}

try:
    print(f"Fetching from {url}...")
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    releases = resp.json()
    
    print("\n=== DETAILED DOWNLOAD STATS ===")
    for release in releases:
        tag = release.get('tag_name', 'Unknown')
        published = release.get('published_at', 'Unknown').split('T')[0]
        
        print(f"\n[{published}] {tag}")
        for asset in release.get('assets', []):
            name = asset.get('name', 'Unknown')
            count = asset.get('download_count', 0)
            print(f"  - {name}: {count}")
            
except Exception as e:
    print(f"Error: {e}")
