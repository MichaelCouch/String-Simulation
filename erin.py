from time import sleep

print "Erin, what a cutie!"

def spell(word):
    for character in word:
        print character
        sleep(0.4)

spell("Michael")

spell("Let's go get dinner!")
