"""
   Getting the photos of actors and actresses.
   Around 5000 males and 5000 females.
   Testing on CPU, so network is not to big.
"""

from requests import get
from bs4 import BeautifulSoup
import os


def main():
    # There are only 2 genders.
    genders = ['male', 'female']

    for gender in genders:
        start = 1
        count = 0
        os.chdir('C:\\Users\\milan\\Downloads\\' + gender)
        while start < 5000:
            # Makeing a url for next page of actors/actresses.
            site = 'https://www.imdb.com/search/name?gender=' + gender + '&count=100&start=' + str(start)

            # Getting html.
            response = get(site)

            # Parsing html.
            soup = BeautifulSoup(response.text, 'html.parser')

            # Getting only 'img' tags
            img_tags = soup.find_all('img')
            urls = []

            # Getting urls of images of actors/actresses.
            for img in img_tags:
                if img['alt'] not in ['IMDbPro Menu', 'Go to IMDbPro'] and 'nopicture' not in img['src']:
                    urls.append(img['src'])

            # Downloading an image of actor/actress
            for url in urls:
                with open(str(count) + '_' + gender + '.jpg', 'wb') as f:
                    response = get(url)
                    f.write(response.content)
                    count += 1

            start += 100


if __name__ == '__main__':
    main()
