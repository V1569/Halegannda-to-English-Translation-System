
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Test Text Translation page
        await page.goto('http://127.0.0.1:5000/pages/text-translation.html')
        await page.wait_for_selector('#inputText', timeout=60000)
        await page.fill('#inputText', 'ನಮಸ್ಕಾರ')
        await page.click('button.btn-translate')
        await page.wait_for_selector('#englishOutput:not(:empty)', timeout=60000)
        english_output = await page.inner_text('#englishOutput')
        print(f'Text Translation Page - English Output: {english_output}')

        # Test JSON Translation page
        await page.goto('http://127.0.0.1:5000/pages/json-translation.html')
        await page.wait_for_selector('#inputWord', timeout=60000)
        await page.fill('#inputWord', 'ಪಂಪ')
        await page.click('button.btn-translate')
        await page.wait_for_selector('#translationOutput:not(:empty)', timeout=60000)
        kannada_output = await page.inner_text('#translationOutput')
        print(f'JSON Translation Page - Kannada Output: {kannada_output}')

        await browser.close()

asyncio.run(main())
