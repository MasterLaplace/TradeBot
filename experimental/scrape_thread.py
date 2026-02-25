import requests
import json
import time
from collections import defaultdict, deque
import re

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


class GraphQLError(Exception):
    def __init__(self, status_code, body, response=None):
        super().__init__(f"GraphQL failed: {status_code}")
        self.status_code = status_code
        self.body = body
        self.response = response

# -------------------------------
# 1) Récupérer le Bearer Token
# -------------------------------

def fetch_public_bearer():
    """
    Récupère automatiquement le Bearer Token public utilisé par X.
    On récupère le code JS principal, et on extrait le Bearer depuis le bundle.
    """
    print("[+] Fetching Bearer token…")

    # 1) charger la home pour trouver le bundle JS
    r = requests.get("https://x.com", headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise Exception("Failed to load X homepage")

    # On cherche le fichier main.*.js
    import re
    bundles = re.findall(r'https://abs\.twimg\.com/responsive-web/client-web/(main\.[^"]+\.js)', r.text)
    if not bundles:
        raise Exception("Unable to find JS bundle containing Bearer token")

    js_url = bundles[0]
    print("[+] JS bundle found:", js_url)

    # 2) télécharger le JS
    js_url = "https://abs.twimg.com/responsive-web/client-web/" + bundles[0]
    r_js = requests.get(js_url, headers={"User-Agent": USER_AGENT})
    js = r_js.text
    # Save the JS locally to ease manual inspection when debugging
    try:
        with open('/tmp/twitter_main_js.txt', 'w', encoding='utf8') as _f:
            _f.write(js)
        print('[+] Saved JS bundle to /tmp/twitter_main_js.txt for inspection')
    except Exception:
        pass

    # 3) extraire le Bearer token
    m = re.search(r'"Bearer ([A-Za-z0-9%-_=]+)"', js)
    if not m:
        raise Exception("Unable to extract Bearer token")

    bearer = m.group(1)
    print("[✓] Bearer token extracted.")
    # Return both bearer and the JS text to allow resolving queryIds dynamically
    return bearer, js


def find_graphql_query_id(js_text, operation_key="threaded_conversation_with_injections_v2"):
    """
    Try to find a stable GraphQL `queryId` (resolver id) for the given operation key
    inside the JS bundle. We look for occurrences of the operation_key then search
    locally around that occurrence for a `queryId` string.
    """
    import re

    # Find candidate positions of the op key
    positions = [m.start() for m in re.finditer(re.escape(operation_key), js_text, flags=re.IGNORECASE)]
    if not positions:
        return None

    # Search around the operation occurrence for a queryId pattern
    for pos in positions:
        window_start = max(0, pos - 2000)
        window_end = pos + 2000
        window = js_text[window_start:window_end]

        # queryId can be expressed as "queryId":"abc123" or as 'queryId':"abc123"
        m = re.search(r"queryId\s*[:=]\s*['\"]([A-Za-z0-9_\-]+)['\"]", window)
        if m:
            return m.group(1)

    return None


def find_graphql_query_ids(js_text, operation_keys=None, max_candidates=10):
    """
    Return a list of candidate queryId strings by searching the JS bundle for provided
    operation keys and falling back to any `queryId` occurrences.
    """
    import re

    if operation_keys is None:
        operation_keys = [
            "threaded_conversation_with_injections_v2",
            "ThreadedConversationWithInjections",
            "TweetDetail",
            "tweet_detail",
            "conversation_with_injections",
        ]

    seen = []
    # Search using operator keys
    for op in operation_keys:
        for match in re.finditer(re.escape(op), js_text, flags=re.IGNORECASE):
            window_start = max(0, match.start() - 2000)
            window_end = match.end() + 2000
            window = js_text[window_start:window_end]
            for m in re.finditer(r"queryId\s*[:=]\s*['\"]([A-Za-z0-9_\-]+)['\"]", window):
                q = m.group(1)
                if q not in seen:
                    seen.append(q)
                    if len(seen) >= max_candidates:
                        return seen

    # Fallback: extract all queryId occurrences from the whole bundle
    for m in re.finditer(r"queryId\s*[:=]\s*['\"]([A-Za-z0-9_\-]+)['\"]", js_text):
        q = m.group(1)
        if q not in seen:
            seen.append(q)
            if len(seen) >= max_candidates:
                return seen

    return seen


def find_valid_query_id_via_probe(bearer, guest_token, js_text, focal_tweet_id, max_tests=5):
    """
    Probe candidate queryIds against the GraphQL endpoint to find one that returns a
    valid JSON payload for the conversation (must contain threaded_conversation... key).
    """
    candidates = find_graphql_query_ids(js_text, max_candidates=max_tests)
    if not candidates:
        # save no-candidates situation for debugging
        try:
            with open('/tmp/twitter_candidates.txt', 'w', encoding='utf8') as f:
                f.write('No candidates found')
        except Exception:
            pass
        return None

    for q in candidates:
        try:
            vars_ = {
                "focalTweetId": focal_tweet_id,
                "withCommunity": False,
                "withVoice": True,
                "includePromotedContent": False,
            }
            resp = x_api(bearer, guest_token, q, vars_)
            if isinstance(resp, dict) and "data" in resp:
                if resp.get("data", {}).get("threaded_conversation_with_injections_v2"):
                    print(f"[✓] Found valid GraphQL id: {q}")
                    return q
        except Exception:
            # On failure, skip and try the next candidate
            continue
    # If none of the probed candidates succeeded, write the candidates to a file
    try:
        with open('/tmp/twitter_candidates.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(candidates))
        print('[!] Probing did not find a working query id; candidates written to /tmp/twitter_candidates.txt')
    except Exception:
        pass

    return None


# -------------------------------
# 2) Récupérer un Guest Token
# -------------------------------

def fetch_guest_token(bearer):
    print("[+] Fetching guest token…")
    r = requests.post(
        "https://api.twitter.com/1.1/guest/activate.json",
        headers={
            "Authorization": f"Bearer {bearer}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
    )
    if r.status_code != 200:
        raise Exception("Failed to obtain guest token")

    guest_token = r.json()["guest_token"]
    print("[✓] Guest token =", guest_token)
    return guest_token


# -------------------------------
# 3) Appel GraphQL (POST)
# -------------------------------

def x_api(bearer, guest_token, query_id, variables):
    r = requests.post(
        f"https://api.twitter.com/graphql/{query_id}",
        headers={
            "Authorization": f"Bearer {bearer}",
            "X-Guest-Token": guest_token,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
        json={"variables": variables}
    )

    if r.status_code != 200:
        # Log details for debugging
        print("ERROR: GraphQL request failed")
        print("URL:", r.url)
        print("Status code:", r.status_code)
        print("Response headers:", r.headers)
        try:
            print("Response body:", r.text)
        except Exception:
            print("Response body: <unable to decode>")
        # Include response body in exception to allow caller to probe 'required features'
        body = r.text
        raise GraphQLError(r.status_code, body, r)

    return r.json()


def timeline_conversation(bearer, guest_token, tweet_id):
    """
    Fallback to the v2 chronology endpoint which returns a timeline JSON for a conversation.
    """
    url = f"https://api.twitter.com/2/timeline/conversation/{tweet_id}.json"
    r = requests.get(url, headers={
        "Authorization": f"Bearer {bearer}",
        "X-Guest-Token": guest_token,
        "User-Agent": USER_AGENT,
    })

    if r.status_code != 200:
        print("ERROR: timeline conversation request failed", r.status_code)
        try:
            print("Response body:", r.text)
        except Exception:
            pass
        raise Exception("Timeline conversation failed")

    return r.json()


# -------------------------------
# 4) Extraction de l'arbre complet
# -------------------------------

def extract_tree(bearer, guest_token, focal_tweet_id, js_text=None):
    """
    On récupère l'arbre complet en BFS.
    """
    print("[+] Scraping tree…")
def extract_tree(bearer, guest_token, focal_tweet_id, js_text=None, preferred_query_id=None):

    queue = deque([focal_tweet_id])
    visited = set()
    tree = defaultdict(lambda: {"tweet": None, "replies": []})

    # Try to find the query id dynamically. If js_text is passed we search in the bundle,
    # otherwise we fall back to the previous known constant (may be outdated).
    # L'identifiant du resolver GraphQL TweetDetail (previously hardcoded)
    FALLBACK_TWEET_DETAIL_ID = "5WcyJLW8N9Zqv5YGHGosgA"

    if js_text:
        TWEET_DETAIL_ID = find_graphql_query_id(js_text, "threaded_conversation_with_injections_v2")
        if not TWEET_DETAIL_ID:
            print("[!] Could not find static operation-linked GraphQL query id in the bundle; probing candidates…")
            q = find_valid_query_id_via_probe(bearer, guest_token, js_text, focal_tweet_id, max_tests=6)
            if q:
                TWEET_DETAIL_ID = q
            else:
                print("[!] Probing failed; falling back to known id.")
                TWEET_DETAIL_ID = FALLBACK_TWEET_DETAIL_ID
    else:
        TWEET_DETAIL_ID = FALLBACK_TWEET_DETAIL_ID

    while queue:
        current = queue.popleft()
        if current in visited:
            continue

        visited.add(current)

        variables = {
            "focalTweetId": current,
            "withCommunity": False,
            "withVoice": True,
            "includePromotedContent": False
        }

        use_timeline_fallback = False
        try:
            data = x_api(bearer, guest_token, TWEET_DETAIL_ID, variables)
        except GraphQLError as e:
            # GraphQL returned something (400/404/etc). If server complains about missing
            # feature flags (400), parse the list and retry with all reported features set to False.
            if e.status_code == 400 and 'cannot be null' in (e.body or ''):
                print('[!] GraphQL returned 400; parsing required features and retrying with defaults')
                m = re.search(r'cannot be null:\s*(.*?)"', e.body, flags=re.DOTALL)
                features_map = {}
                if m:
                    raw = m.group(1)
                    # split by comma and map names to False
                    for name in re.split(r',\s*', raw):
                        name = name.strip()
                        if name:
                            features_map[name] = False
                if features_map:
                    new_vars = dict(variables)
                    new_vars['features'] = features_map
                    try:
                        data = x_api(bearer, guest_token, TWEET_DETAIL_ID, new_vars)
                    except Exception:
                        print('[!] Retry with default features failed')
                        # Fallthrough to timeline fallback
                else:
                    print('[!] Could not parse features list; falling back to timeline endpoint')

            # If we reach here without data, try timeline fallback
            if not 'data' in locals():
                print(f"GraphQL request with id {TWEET_DETAIL_ID} failed; trying timeline endpoint as fallback.")
                try:
                    data = timeline_conversation(bearer, guest_token, current)
                    use_timeline_fallback = True
                except Exception:
                    print("Timeline fallback also failed for tweet", current)
                    raise

        if not use_timeline_fallback:
            instructions = (
                data.get("data", {})
                    .get("threaded_conversation_with_injections_v2", {})
                    .get("instructions", [])
            )
        else:
            # timeline v2 JSON format provides a list of tweets in `globalObjects.tweets`.
            instructions = None

        for instr in instructions:
            if instr.get("type") == "TimelineAddEntries":
                for entry in instr.get("entries", []):
                    content = entry.get("content", {})
                    item = content.get("item", {})
                    tweet_results = item.get("tweet_results")

                    if not tweet_results:
                        continue

                    result = tweet_results.get("result")
                    if not result:
                        continue

                    tweet_id = result.get("rest_id")
                    if not tweet_id:
                        continue

                    # Enregistrer le tweet
                    tree[tweet_id]["tweet"] = result

                    # Explorer les réponses attachées
                    replies = item.get("replies", [])
                    for rep in replies:
                        rep_id = rep["tweet"]["id"]
                        tree[tweet_id]["replies"].append(rep_id)
                        queue.append(rep_id)

        if use_timeline_fallback:
            # parse timeline JSON globalObjects.tweets (flat list) and build relations
            tweets = data.get('globalObjects', {}).get('tweets', {})
            for tid, tweet in tweets.items():
                tree[tid]['tweet'] = tweet
                in_reply_id = tweet.get('in_reply_to_status_id')
                if in_reply_id:
                    tree[in_reply_id]['replies'].append(tid)
                    if tid not in visited:
                        queue.append(tid)

        time.sleep(0.2)  # éviter rate-limit

    print("[✓] Full tree scraped.")
    return tree


# -------------------------------
# 5) Main
# -------------------------------

def extract_tweet_id(url):
    return url.rstrip("/").split("/")[-1]


def scrape_thread(url, preferred_query_id=None):
    focal_id = extract_tweet_id(url)
    bearer, js = fetch_public_bearer()
    guest_token = fetch_guest_token(bearer)

    tree = extract_tree(bearer, guest_token, focal_id, js_text=js, preferred_query_id=preferred_query_id)

    with open("thread_output.json", "w", encoding="utf8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print("[✓] Saved to thread_output.json")
    return tree


if __name__ == "__main__":
    import sys
    URL = sys.argv[1] if len(sys.argv) > 1 else "https://x.com/Saiji_Tv/status/1951321878590538072"
    preferred_qid = sys.argv[2] if len(sys.argv) > 2 else None
    scrape_thread(URL, preferred_query_id=preferred_qid)
