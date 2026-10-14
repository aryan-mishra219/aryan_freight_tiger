import os
import sys
import zipfile
from pathlib import Path

def create_submission_zip():
    project_root = Path(__file__).resolve().parent
    dest_zip_path = project_root.parent / "Aryan_FreightTiger_Submission_Clean.zip"
    
    # Remove existing zip if present
    if dest_zip_path.exists():
        dest_zip_path.unlink()
        
    print(f"Packaging project from: {project_root}")
    print(f"Target submission archive: {dest_zip_path}")
    
    excluded_names = {".env"}
    excluded_patterns = ["__pycache__", ".git", ".venv", "venv"]
    
    included_files = []
    
    with zipfile.ZipFile(dest_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in sorted(project_root.rglob("*")):
            if not file_path.is_file():
                continue
                
            rel_path = file_path.relative_to(project_root)
            parts = rel_path.parts
            
            # Exclusion rules
            if file_path.name in excluded_names:
                print(f"  [EXCLUDED STRICTLY]: {rel_path} (Private Secret)")
                continue
                
            if any(p in parts for p in excluded_patterns):
                continue
                
            if file_path.name.endswith(".pyc"):
                continue
                
            # Exclude this packaging script itself if desired, or include it
            if file_path.name == "package_submission.py":
                continue
                
            # Archive relative path with forward slashes inside zip
            archive_name = f"freight_tiger_case_study/{rel_path.as_posix()}"
            zipf.write(file_path, arcname=archive_name)
            included_files.append(archive_name)

    print("\n" + "=" * 70)
    print("SUBMISSION ARCHIVE VERIFICATION AUDIT")
    print("=" * 70)
    
    # Re-open and verify contents
    with zipfile.ZipFile(dest_zip_path, "r") as zipf:
        entries = zipf.namelist()
        
    print(f"Total entries in archive: {len(entries)}")
    print(f"Archive size: {dest_zip_path.stat().st_size / 1024:.1f} KB\n")
    
    # Assertions
    has_secret_env = any(entry.endswith("/.env") or entry == ".env" for entry in entries)
    has_env_example = any(".env.example" in entry for entry in entries)
    has_pycache = any("__pycache__" in entry or entry.endswith(".pyc") for entry in entries)
    has_git = any(".git/" in entry or entry.endswith(".git") for entry in entries)
    
    assert not has_secret_env, "FATAL SECURITY ERROR: .env was found in the archive!"
    assert has_env_example, "ERROR: .env.example was not found in the archive!"
    assert not has_pycache, "ERROR: __pycache__ or .pyc found in the archive!"
    assert not has_git, "ERROR: .git directory found in the archive!"
    
    print("Security & Cleanliness Invariants:")
    print("  [x] .env strictly EXCLUDED:       VERIFIED PASSED")
    print("  [x] .env.example INCLUDED:        VERIFIED PASSED")
    print("  [x] __pycache__ & *.pyc EXCLUDED: VERIFIED PASSED")
    print("  [x] .git & .venv EXCLUDED:        VERIFIED PASSED")
    print("\nArchived File Manifest:")
    for entry in sorted(entries):
        print(f"  {entry}")
        
    print("=" * 70)
    print(f"SUCCESS: Submission zip verified and saved to:\n  {dest_zip_path}")
    print("=" * 70)

if __name__ == "__main__":
    create_submission_zip()
