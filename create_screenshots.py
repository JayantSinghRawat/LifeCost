import asyncio
from playwright.async_api import async_playwright
import os

os.makedirs('artifacts', exist_ok=True)

async def capture_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', 
            headless=True
        )
        page = await browser.new_page(viewport={"width": 1400, "height": 1000})

        url = 'file:///Users/jayant/Desktop/MP-Life/frontend/index.html'
        print(f"Loading {url}...")
        
        await page.goto(url, wait_until='networkidle')

        try:
            buttons = await page.get_by_text("Predict", exact=False).element_handles()
            for btn in buttons:
                await btn.click()
            
            calc_btn = await page.get_by_text("Calculate", exact=False).element_handles()
            for btn in calc_btn:
                await btn.click()
                
            gen_btn = await page.get_by_text("Generate", exact=False).element_handles()
            for btn in gen_btn:
                await btn.click()
        except:
            pass

        await page.wait_for_timeout(3000)

        # 7.1 Landing Page
        await page.screenshot(path='/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_1_Landing_Page.png', full_page=False)
        print("Captured 7.1")

        async def snap(selector, filename, fallback_scroll):
            el = await page.query_selector(selector)
            if el:
                await el.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)
                await el.screenshot(path=filename)
            else:
                await page.evaluate(f"window.scrollTo(0, {fallback_scroll})")
                await page.screenshot(path=filename)

        await snap('.predict-panel', '/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_2_Rent_Prediction.png', 500)
        print("Captured 7.2")

        await snap('.chart-card', '/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_3_CPI_Chart.png', 1000)
        print("Captured 7.3")

        el = await page.query_selector('.leaflet-container')
        if not el: el = await page.query_selector('#map')
        if el:
            await el.scroll_into_view_if_needed()
            await page.wait_for_timeout(1500)
            await el.screenshot(path='/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_4_Locality_Map.png')
        else:
            await page.evaluate("window.scrollTo(0, 1500)")
            await page.screenshot(path='/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_4_Locality_Map.png')
        print("Captured 7.4")

        await snap('.metro-calc', '/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_5_Metro_Fare.png', 2000)
        print("Captured 7.5")

        await snap('.gauge-section', '/Users/jayant/Desktop/MP-Life/artifacts/Screenshot_7_6_Cost_Of_Living.png', 2500)
        print("Captured 7.6")

        await browser.close()
        print("Done capturing 6 screenshots!")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
