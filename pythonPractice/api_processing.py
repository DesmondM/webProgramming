import requests


def main():
    print("Search the Art institute of Chicago!")
    artist = input("Artist: ")
    try:
        response= requests.get('https://api.artic.edu/api/v1/artworks/search', {"q":artist})
    except requests.HTTPError:
        print("Couldn't complete request")
    content = response.json()

    for artwork in content['data']:
        print(f"- {artwork['id']} ## {artwork['title']}")

main()