import urllib
from urllib import request
import json
import time
import pandas as pd
import csv

hdr = {'User-Agent': 'r/getdisciplined/ (by /u/<username>)'} #put your username here
url = 'https://www.reddit.com/r/getdisciplined/hot/.json?sort=top&t=all&limit=100'
req = request.Request(url, headers=hdr)
text_data = urllib.request.urlopen(req).read()
data = json.loads(text_data)
vlist = list(data.values())
post_list = vlist[1]['children']

while len(post_list) <= 700:
    time.sleep(2)
    last = post_list[-1]['data']['name']
    url = 'https://www.reddit.com/r/getdisciplined/hot/.json?sort=top&t=all&limit=100&after=%s' % last
    req = request.Request(url, headers=hdr)
    text_data = urllib.request.urlopen(req).read()
    data = json.loads(text_data)
    post_list += list(data.values())[1]['children']
    print(len(post_list))

filtered_post_list = []
for i in range(len(post_list)):
    post_title = post_list[i]['data']['title'].lower()
    if 'need advice' in post_title:
        filtered_post_list.append(post_list[i])
    elif 'needadvice' in post_title:
        filtered_post_list.append(post_list[i])
    else:
        pass
print(len(filtered_post_list))

filtered_sorted = sorted(filtered_post_list, key=lambda k: k['data']['score'], reverse=True)

with open('5.1._Preliminary_studies/raw_dataset.csv', 'a') as f:
    fieldnames = ['Title', 'Post_Text', 'Post_Score', 'Id', 'Coms_no', 'Com_text', 'Com_score']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    for j in range(2, len(filtered_sorted)):
        post_dane = filtered_sorted[j]['data']
        post_title = post_dane['title']
        post_text = post_dane['selftext']
        post_score = int(post_dane['score'])
        post_unique_id = post_dane['name']
        post_permalink = post_dane['permalink']
        link = ''.join('https://www.reddit.com') + ''.join(post_permalink) + ''.join('.json')
        req_coms = request.Request(url=link, headers=hdr)
        text_coms = urllib.request.urlopen(req_coms).read()
        comdata = json.loads(text_coms)
        coms = comdata[1]['data']['children']
        if (post_score > 0) and (len(coms) > 0):
            for x in range(len(coms)):
                writer.writerow({'Title': post_title, 'Post_Text': post_text, 'Post_Score': post_score,
                             'Id': post_unique_id, 'Coms_no': len(coms), 'Com_text': coms[x]['data']['body'], 'Com_score':coms[x]['data']['score']})
