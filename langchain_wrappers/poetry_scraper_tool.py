import tempfile
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.tools import BaseTool


class PoetryScraperTool(BaseTool):
    name: str = "PoetryScraperTool"
    description: str = "Scrapes Poetry Foundation for a poem given a title."

    def _run(self, title: str) -> dict:
        # Configure Chrome options
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")  # Unique temp directory

        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        try:
            query = f"{title} poetry foundation".replace(' ','+')
            print(query)
            driver.get(f"https://www.google.com/search?q={query}")

            # Wait for the search input and button to appear
            wait = WebDriverWait(driver, 10)
            inp = wait.until(EC.presence_of_element_located(
                (By.XPATH, '//*[@id="__nuxt"]/div/div/div[2]/header[1]/div/div/div[1]/form/div/input')
            ))
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="__nuxt"]/div/div/div[2]/header[1]/div/div/div[1]/form/div/button')
            ))
            inp.send_keys(title)
            button.click()

            # Wait for the list of poems to load and then click the first poem link
            poem_link = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="mainContent"]/div/div[3]/div[1]/div/article[1]/div/a')
            ))
            poem_link.click()

            # Wait for the poem page to load by checking for the poem body element
            wait.until(EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'poem-body')]")
            ))

            # Parse the page source
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            poet = soup.find('a', class_='link-underline-off link-red')
            poem_body = soup.find('div', class_='poem-body')
            poem = {
                'poet': poet.text.strip() if poet else 'Unknown',
                'body': poem_body.text.strip() if poem_body else 'No content available',
            }
        except Exception as e:
            poem = {"error": str(e)}
        finally:
            driver.quit()

        return poem

    async def _arun(self, title: str) -> dict:
        # Asynchronous execution is not implemented in this example.
        raise NotImplementedError("Asynchronous run is not supported.")
