# Injection script that will save results from a running ok.py process.
# Usage: pyrasite-ng <PID> save_now.py

import gc, json, os

out = '/tmp/ok_results_injected.json'
found = None
for obj in gc.get_objects():
    try:
        if getattr(obj, '__class__', None) and obj.__class__.__name__ == 'XScraper':
            found = obj
            break
    except Exception:
        continue

if not found:
    # fallback: try to detect a variable named scraper in globals
    g = globals()
    if 'scraper' in g and hasattr(g['scraper'], 'results'):
        found = g['scraper']

if not found:
    print('No XScraper instance found in process')
else:
    print('Found XScraper, saving results count:', len(getattr(found, 'results', [])))
    try:
        results = getattr(found, 'results', [])
        with open(out + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(out + '.tmp', out)
        print('Saved to', out)
    except Exception as e:
        print('Failed to save results:', e)
