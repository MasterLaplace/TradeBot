import asyncio
import json
import os
import sys
import time
import signal
import atexit
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

USERNAME = os.getenv("X_USERNAME")
PASSWORD = os.getenv("X_PASSWORD")

COOKIE_FILE = "cookies.json"

class XScraper:
    def __init__(self, tweet_url, output_file: str = "results.json", auto_save_every: int = 20):
        self.tweet_url = tweet_url
        self.visited = set()
        self.results = []
        self.output_file = output_file
        # auto-save after this many appended items
        self.auto_save_every = auto_save_every
        self._items_since_save = 0
        self._last_save_time = 0
        # On se charge de reprendre si un fichier de résultats existe déjà
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    self.results = existing
                    # marquer tous les URLs déjà traités
                    for item in self.results:
                        url = item.get('url') if isinstance(item, dict) else None
                        if url:
                            self.visited.add(url)
                print(f"[RESUME] Loaded {len(self.results)} existing results from {self.output_file}")
        except Exception as e:
            print('[WARN] Unable to load existing output file for resume:', e)

    async def login_if_needed(self, context, page):
        """Se connecte uniquement si les cookies n'existent pas."""
        if not os.path.exists(COOKIE_FILE):
            print("[LOGIN] Aucun cookie trouvé → Connexion à X…")

            await page.goto("https://x.com/login", timeout=150000)

            # Champs login
            await page.fill("input[name='text']", USERNAME)
            await page.click("text=Next")

            await page.wait_for_selector("input[name='password']", timeout=20000)
            await page.fill("input[name='password']", PASSWORD)
            await page.click("text=Log in")

            # Attendre la page d'accueil
            await page.wait_for_selector("nav", timeout=30000)

            # Sauvegarde des cookies
            cookies = await context.cookies()
            with open(COOKIE_FILE, "w") as f:
                json.dump(cookies, f, indent=2)

            print("[LOGIN] Connexion réussie et cookies sauvegardés.")
        else:
            print("[LOGIN] Cookies trouvés → Chargement…")
            with open(COOKIE_FILE, "r") as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
            print("[LOGIN] Cookies chargés.")

    async def run(self):
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=False)
            context = await browser.new_context()

            page = await context.new_page()

            await self.login_if_needed(context, page)

            # Maintenant qu'on est login, on peut scraper
            await self.scrape_thread(page, self.tweet_url)

            # Save results on normal completion as well
            await self._save_results()

            print(f"[OK] Scraping terminé → {len(self.results)} messages collectés.")

            # Sauvegarde des cookies (au cas où X en régénère)
            cookies = await context.cookies()
            with open(COOKIE_FILE, "w") as f:
                json.dump(cookies, f, indent=2)

            await browser.close()

    def _install_exit_handlers(self):
        """Install signal handlers for graceful shutdown and atexit hook.
        """
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, lambda: asyncio.create_task(self._on_exit()))
                except NotImplementedError:
                    # Windows or non-main thread may raise
                    signal.signal(sig, lambda *_: asyncio.run(self._on_exit()))
        except RuntimeError:
            # No running loop yet — fallback
            signal.signal(signal.SIGTERM, lambda *_: asyncio.run(self._on_exit()))
            signal.signal(signal.SIGINT, lambda *_: asyncio.run(self._on_exit()))

        # Also ensure at-exit we attempt to save
        atexit.register(lambda: asyncio.run(self._on_exit()))

    async def _save_results(self):
        """Write self.results to disk atomically. Use a tmp file and os.replace.
        This method is async-friendly (non-blocking minimal I/O).
        """
        try:
            tmp = Path(self.output_file + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            os.replace(str(tmp), self.output_file)
            self._items_since_save = 0
            self._last_save_time = time.time()
            print(f"[SAVE] Checkpoint: {len(self.results)} messages saved to {self.output_file}")
        except Exception as e:
            print("[ERROR] Failed to save results:", e)

    def _maybe_schedule_save(self):
        """Decide if we should save now; called synchronously from scraping logic.
        Schedules the async save on the loop if needed.
        """
        self._items_since_save += 1
        # Save either every N items, or every 60 seconds
        if self._items_since_save >= self.auto_save_every or (time.time() - self._last_save_time > 60):
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._save_results())
            except RuntimeError:
                # Not in an event loop (should not happen here), fallback to sync save
                try:
                    self._last_save_time = time.time()
                    with open(self.output_file, "w", encoding="utf-8") as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print("[ERROR] Failed to save results (sync fallback):", e)

    # -----------------------------------------------------
    # SCRAPE DFS COMPLET DU THREAD
    # -----------------------------------------------------
    async def scrape_thread(self, page, url):
        if url in self.visited:
            return
        self.visited.add(url)

        print(f"[SCRAPE] {url}")

        await page.goto(url, timeout=150000)
        await asyncio.sleep(4)

        # Scroll agressif pour charger toutes les réponses
        for _ in range(15):
            await page.mouse.wheel(0, 5000)
            await asyncio.sleep(0.8)

        tweets = await page.locator("article").all()

        thread_links = []

        for tw in tweets:
            try:
                # Extraction du lien du tweet
                links = await tw.locator("a[href*='/status/']").evaluate_all(
                    "els => els.map(e => e.href)"
                )

                tweet_url = next((h for h in links if "/status/" in h), None)
                if not tweet_url:
                    continue

                text = (await tw.inner_text()).strip()

                self.results.append({
                    "url": tweet_url,
                    "text": text
                })

                # Save progress incrementally so we don't lose results
                self._maybe_schedule_save()

                thread_links.append(tweet_url)

            except Exception as e:
                print("[WARN] Tweet illisible :", e)

        # DFS récursif
        for link in thread_links:
            if link not in self.visited:
                await self.scrape_thread(page, link)

    # Support clean exit on signals
    async def _on_exit(self):
        print("[EXIT] Signal received: saving results before exit...")
        await self._save_results()
        print("[EXIT] Saved. Exiting now.")
        # ensure finally flush
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# -----------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------
if __name__ == "__main__":
    url = os.getenv("TWEET_URL") or "https://x.com/Saiji_Tv/status/1951321878590538072"
    out = os.getenv("OUTPUT_FILE") or "results.json"
    save_every = int(os.getenv("SAVE_EVERY", "20"))
    scraper = XScraper(url, output_file=out, auto_save_every=save_every)
    scraper._install_exit_handlers()
    asyncio.run(scraper.run())
