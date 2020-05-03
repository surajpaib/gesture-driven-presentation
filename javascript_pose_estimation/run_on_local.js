const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        args: [
            "—-use-gl=egl",
            '--use-fake-device-for-media-stream',
            '--use-file-for-fake-video-capture=/Users/admin/Desktop/test.y4m'
        ],
    });
    const page = await browser.newPage();
    await page.goto('http://localhost:7777');
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));

    await page.evaluate(() => console.log(`url is ${location.href}`));

    // await browser.close();
})();