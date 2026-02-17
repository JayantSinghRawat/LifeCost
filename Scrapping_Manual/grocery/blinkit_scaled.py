import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
PINCODE = "462010"
SEARCH_TERMS = ["milk", "vegetables", "fruits", "bread", "eggs"]
OUTPUT_FILE = "blinkit_results_462010.json"

def init_driver():
    """Initializes the Firefox WebDriver."""
    print("Initializing Firefox Driver...")
    options = webdriver.FirefoxOptions()
    # options.add_argument("--headless") # Uncomment if you don't want to see the browser
    driver = webdriver.Firefox(options=options)
    driver.maximize_window()
    return driver

def set_location(driver, pincode):
    """Sets the delivery location pincode."""
    print(f"Setting location to {pincode}...")
    try:
        # Wait for the location input to be clickable
        wait = WebDriverWait(driver, 15)
        
        # Sometimes there's a different initial location or 'detect location' prompt
        # We try to find the input box directly
        location_box = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[@placeholder="search delivery location"]')))
        location_box.click()
        
        # Clear any existing text just in case
        location_box.clear() 
        location_box.send_keys(pincode)
        time.sleep(2) # Wait for suggestions
        
        # Click the suggestion that matches the pincode
        location_option = wait.until(EC.element_to_be_clickable((By.XPATH, f'//div[contains(@class,"LocationSearchList__LocationLabel") and text()="{pincode}"]')))
        location_option.click()
        
        print("Location set successfully.")
        time.sleep(3) # Wait for page reload/update
        return True
    except Exception as e:
        print(f"Error setting location: {e}")
        return False

def scroll_to_bottom(driver):
    """Scrolls to the bottom of the page to load all products."""
    print("Scrolling to load all products...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3) # Wait for content to load
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # Try scrolling up a bit and back down to trigger load if stuck
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 500);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
        last_height = new_height
    print("Finished scrolling.")

def scrape_products(driver, search_term):
    """Searches for a term and scrapes product details."""
    print(f"Searching for: {search_term}")
    product_data = []
    
    try:
        wait = WebDriverWait(driver, 10)
        
        # Find search bar (handling both collapsed and expanded states)
        try:
            search_trigger = driver.find_element(By.XPATH, '//div[contains(@class,"SearchBar__AnimationWrapper")]')
            search_trigger.click()
        except:
            pass # Search might already be active/different layout
            
        search_input = wait.until(EC.presence_of_element_located((By.XPATH, '//input[contains(@class,"SearchBarContainer__Input")]')))
        
        # Clear utilizing keys because standard .clear() can be flaky locally with React apps
        search_input.send_keys(Keys.COMMAND + "a")
        search_input.send_keys(Keys.DELETE)
        search_input.send_keys(search_term)
        search_input.send_keys(Keys.RETURN)
        
        time.sleep(5) # Wait for initial results
        
        # Infinite Scroll
        scroll_to_bottom(driver)
        
        # Extract Data
        # Using a more robust selector for product cards
        product_cards = driver.find_elements(By.XPATH, '//div[@data-test-id="product-card-container"] | //div[contains(@class,"Product__UpdatedContentContainer")]/..')
        
        # Fallback if specific test id is missing, try the generic grid item
        if not product_cards:
             product_cards = driver.find_elements(By.XPATH, '//div[contains(@class,"categories-table")]/div/div')

        print(f"Found {len(product_cards)} items for '{search_term}'. Extracting details...")
        
        for card in product_cards:
            try:
                text_content = card.text.split('\n')
                # Basic heuristic parsing based on typical Blinkit card structure
                # Usually: [Delivery Time, Name, Quantity, Price, ADD button]
                
                # Filter out "ADD" and empty strings
                clean_text = [t for t in text_content if t and t.upper() != "ADD"]
                
                if len(clean_text) >= 3:
                    item = {
                        "term": search_term,
                        "raw_text": clean_text
                    }
                    
                    # Try to identify specific fields
                    # This is fragile and depends on UI, but scaleable logic attempts to guess
                    item['name'] = clean_text[1] if len(clean_text) > 1 else "Unknown"
                    item['quantity'] = clean_text[2] if len(clean_text) > 2 else "Unknown"
                    item['price'] = clean_text[3] if len(clean_text) > 3 else "Unknown"
                    
                    # Correction: usually element 0 is delivery time ("8 MINS")
                    if "MINS" in clean_text[0].upper():
                        item['delivery_time'] = clean_text[0]
                    else:
                        item['name'] = clean_text[0] # Shift if no delivery time
                    
                    product_data.append(item)
            except Exception as e:
                continue # Skip broken card
                
    except Exception as e:
        print(f"Error scraping term '{search_term}': {e}")
        
    return product_data

def main():
    driver = init_driver()
    all_products = []
    
    try:
        driver.get('https://blinkit.com/')
        time.sleep(5)
        
        if set_location(driver, PINCODE):
            for term in SEARCH_TERMS:
                products = scrape_products(driver, term)
                all_products.extend(products)
                time.sleep(2) # Pause between searches
        
        # Save to file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=4, ensure_ascii=False)
        print(f"Scraping complete. Data saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
