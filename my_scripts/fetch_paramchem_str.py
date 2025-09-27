#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Best-effort automation to upload a ligand .mol2 to ParamChem (CGenFF),
then download the resulting .str file.

Notes/limits:
- ParamChem is a web service that may change UI, require login or human steps.
- This script uses Selenium + webdriver-manager. If automation fails,
  it opens a browser window for you to finish steps manually; downloads will
  be saved to the specified directory.

Usage examples (PowerShell):
  # With interactive browser (recommended)
  python scripts/fetch_paramchem_str.py --mol2 out/gmx_split_20250924_011827/hiv.mol2 --out out/gmx_split_20250924_011827

  # Provide credentials via env (optional, if you have an account)
  $env:PARAMCHEM_EMAIL = 'you@example.com'
  $env:PARAMCHEM_PASSWORD = 'secret'
  python scripts/fetch_paramchem_str.py --mol2 path/to/ligand.mol2 --out outdir
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except Exception as e:
    print("Selenium/webdriver-manager not available. Install with:\n  pip install selenium webdriver-manager", file=sys.stderr)
    raise


PARAMCHEM_URL = "https://cgenff.paramchem.org/"


def build_driver(download_dir: Path, headless: bool = False) -> webdriver.Chrome:
    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_window_size(1280, 900)
    return driver


def try_login(driver, email: str | None, password: str | None):
    if not email or not password:
        return False
    try:
        # Heuristic: look for a login form
        time.sleep(1)
        # Possible login triggers
        # find email input
        email_el = None
        for sel in ('input[type="email"]', 'input[name*="email"]', 'input[id*="email"]'):
            try:
                email_el = driver.find_element(By.CSS_SELECTOR, sel)
                if email_el:
                    break
            except Exception:
                continue
        if not email_el:
            return False
        email_el.clear(); email_el.send_keys(email)
        pwd_el = None
        for sel in ('input[type="password"]', 'input[name*="pass"]', 'input[id*="pass"]'):
            try:
                pwd_el = driver.find_element(By.CSS_SELECTOR, sel)
                if pwd_el:
                    break
            except Exception:
                continue
        if not pwd_el:
            return False
        pwd_el.clear(); pwd_el.send_keys(password)
        # click submit
        for sel in ('button[type="submit"]', 'input[type="submit"]', 'button[name*="login"]'):
            try:
                driver.find_element(By.CSS_SELECTOR, sel).click()
                break
            except Exception:
                continue
        time.sleep(2)
        return True
    except Exception:
        return False


def upload_and_download(driver, mol2: Path, timeout_sec: int = 300) -> Path | None:
    wait = WebDriverWait(driver, 20)
    # Try to locate upload input
    file_input = None
    for sel in ('input[type="file"]', 'input[name*="file"]'):  # heuristic
        try:
            file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            if file_input:
                break
        except Exception:
            continue
    if not file_input:
        print("Could not find file upload field. Please complete the upload manually in the opened browser.")
        return None
    file_input.send_keys(str(mol2.resolve()))
    # Click submit
    for sel in ('button[type="submit"]', 'input[type="submit"]', 'button[name*="submit"]', 'button:contains("Submit")'):
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            btn.click()
            break
        except Exception:
            continue
    # Wait for result page and .str link
    t0 = time.time()
    str_path = None
    while time.time() - t0 < timeout_sec:
        # Look for links to .str
        links = driver.find_elements(By.CSS_SELECTOR, 'a[href$=".str"]')
        if links:
            try:
                links[0].click()
            except Exception:
                pass
            # Wait a bit for download
            time.sleep(5)
            break
        time.sleep(2)
    return None


def latest_str_in_dir(download_dir: Path, since: float) -> Path | None:
    cands = list(download_dir.glob("*.str"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if cands[0].stat().st_mtime >= since:
        return cands[0]
    return None


def main():
    ap = argparse.ArgumentParser(description="Fetch .str from ParamChem for a given .mol2")
    ap.add_argument("--mol2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output directory to place the .str copy")
    ap.add_argument("--headless", action="store_true", help="Run browser headless")
    args = ap.parse_args()

    if not args.mol2.exists():
        print(f"Input mol2 not found: {args.mol2}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("PARAMCHEM_EMAIL")
    password = os.environ.get("PARAMCHEM_PASSWORD")

    # Use a temp download dir under out
    dl_dir = (args.out / "paramchem_downloads").resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)

    driver = build_driver(dl_dir, headless=args.headless)
    try:
        print(f"Opening {PARAMCHEM_URL} ...")
        driver.get(PARAMCHEM_URL)
        # Optionally try login
        if email and password:
            try_login(driver, email, password)
        t_start = time.time()
        # Attempt automated upload
        upload_and_download(driver, args.mol2)
        # If automation failed or requires manual steps, allow user to finish manually
        print("If automation didn't complete, please finish upload manually in the opened browser.")
        print("Waiting up to 5 minutes for a .str to appear in the download dir...")
        # Poll for .str file
        deadline = time.time() + 300
        found = None
        while time.time() < deadline:
            found = latest_str_in_dir(dl_dir, since=t_start)
            if found:
                break
            time.sleep(3)
        if not found:
            print(f"No .str file detected in: {dl_dir}", file=sys.stderr)
            return 3
        # Copy to out as <stem>.str
        target = args.out / (args.mol2.stem + ".str")
        target.write_bytes(found.read_bytes())
        print(f"Saved: {target}")
        return 0
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

