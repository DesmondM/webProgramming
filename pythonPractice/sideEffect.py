emoticon = "😊"

def main():
    global emoticon
    say("Hi is there someone there?")
    emoticon = "😢"
    say("am sad today")

    emoticon = "🤔"
    say("Who knows what tomorrow brings?")
    

def say(message):
    print(f"{emoticon} {message}")

    
main()