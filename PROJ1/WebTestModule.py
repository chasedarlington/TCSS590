from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

class WebPageController:
    def __init__(self, url):
        self.url = url
        self.driver = webdriver.Chrome()
        self.actions = ActionChains(self.driver)

    def open_page(self):
        self.driver.get(self.url)
        time.sleep(2)

    def press_left(self):
        self.actions.send_keys(Keys.ARROW_LEFT).perform()

    def press_right(self):
        self.actions.send_keys(Keys.ARROW_RIGHT).perform()

    def press_up(self):
        self.actions.send_keys(Keys.ARROW_UP).perform()

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    url = input("Enter webpage URL: ")

    controller = WebPageController(url)
    controller.open_page()


    print("Press:")
    print("  l = left arrow")
    print("  r = right arrow")
    print("  u = up arrow")
    print("  q = quit")

    while True:
        command = input("Command: ").lower()

        if command == "l":
            controller.press_left()
        elif command == "r":
            controller.press_right()
        elif command == "u":
            controller.press_up()
        elif command == "q":
            controller.close()
            break
        else:
            print("Unknown command")