import urllib
from urllib import request
import json
import time
import pandas as pd
import pickle
import random
import codecs

hdr = {'User-Agent': 'r/todayilearned/ (by /u/<username>)'} #put your username here
url = 'https://www.reddit.com/r/todayilearned/hot/.json?sort=top&t=all&limit=100'
req = request.Request(url, headers=hdr)
text_data = urllib.request.urlopen(req).read()
data = json.loads(text_data)
vlist = list(data.values())
post_list = vlist[1]['children']

while len(post_list) <= 600:
    time.sleep(2)
    last = post_list[-1]['data']['name']
    url = 'https://www.reddit.com/r/todayilearned/hot/.json?sort=top&t=all&limit=100&after=%s' % last
    req = request.Request(url, headers=hdr)
    text_data = urllib.request.urlopen(req).read()
    data = json.loads(text_data)
    post_list += list(data.values())[1]['children']
    print(len(post_list), 'posts retrieved')

randomized = random.sample(post_list, len(post_list))

data = []

for i in range(len(randomized)):
    try:
        post_dane = randomized[i]['data']
        post_text = post_dane['selftext'].encode('utf8').decode('utf8')
        post_title = post_dane['title'].encode('utf8').decode('utf8')
        post_permalink = post_dane['permalink']
        post_score = post_dane['score']
        link = ''.join('https://www.reddit.com') + ''.join(post_permalink) + ''.join('.json')
        req_coms = request.Request(url=link, headers=hdr)
        text_coms = urllib.request.urlopen(req_coms).read()
        comdata = json.loads(text_coms)
        coms_0 = []
        for item in comdata[1]['data']['children']:
            if item['data']['body'] != '':
                coms_0.append(item)
    except (KeyError, UnicodeEncodeError, UnicodeDecodeError):
        pass
    coms = sorted(coms_0, key=lambda k: k['data']['score'], reverse=True)

    for x in range(len(coms)):
        com_text = coms[x]['data']['body'].encode('utf8').decode('utf8')
        com_score = coms[x]['data']['score']
        line = [post_title, post_text, post_score, com_text, com_score]
        data.append(line)

labels = ['Title', 'Post', 'Post_score', 'Comment', 'Comment_score']
df = pd.DataFrame.from_records(data, columns=labels)
df.to_pickle('todayilearned')
print(df.to_string(), file=open('todayilearned.txt', 'w', encoding='utf8'))