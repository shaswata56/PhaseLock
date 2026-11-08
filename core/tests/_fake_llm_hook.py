import json, sys
req = json.load(sys.stdin)
paths = [t['path'] for t in req['library']][::-1]
print(json.dumps(paths))
