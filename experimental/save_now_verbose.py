import gc, json, os, time

logf = '/tmp/pyrasite_dump.log'
def log(msg):
    with open(logf, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(msg) + '\n')

log('Injection started')
found = None
try:
    for o in gc.get_objects():
        try:
            if getattr(o, '__class__', None) and o.__class__.__name__ == 'XScraper':
                found = o
                break
        except Exception:
            continue
    log('search done, found: ' + str(bool(found)))
    if not found and 'scraper' in globals():
        g = globals()
        if 'scraper' in g:
            try:
                if hasattr(g['scraper'], 'results'):
                    found = g['scraper']
            except Exception:
                pass
    if found:
        results = getattr(found, 'results', None)
        log('results type: ' + str(type(results)) + ' len: ' + str(len(results) if results else 0))
        try:
            path = '/tmp/ok_results_injected.json'
            with open(path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            os.replace(path + '.tmp', path)
            log('Saved results to ' + path)
        except Exception as e:
            log('exception writing results: ' + str(e))
    else:
        log('No XScraper found in process')
except Exception as e:
    log('general exception: ' + str(e))
log('Injection completed')
