const
    HeadlessChrome = require('simple-headless-chrome')
process = require('process')
fs = require('fs')
    ;

const browser = new HeadlessChrome({
    headless: false, // If you turn this off, you can actually see the browser navigate with your instructions
    // see above if using remote interface
    chrome: {
        flags: ['--disable-gpu', '--window-size=1280,1696', '--enable-logging'],
        //flags: ['--headless', '--disable-gpu', '--window-size=1280,1696', '--enable-logging'],
        noSandbox: true
    },
    deviceMetrics: {
        //fitWindow: true
    }
})

var address, bet, num, typebet, simulation = false;

var debug = 1;

var config = JSON.parse(fs.readFileSync('./scripts/bet_config.json', 'utf8'));

console.log(JSON.stringify(config, null, 4));

async function play() {
    console.log('init browser');

    await browser.init();

    console.log('open new tab');

    const mainTab = await browser.newTab({ privateTab: false })
    await mainTab.resizeFullScreen()

    mainTab.onConsole(function (msg) {
        console.log(msg)
    })

    console.log('open address');

    await mainTab.goTo(address);
    await mainTab.wait(2000);

    await mainTab.inject('jQuery');

    mainTab.evaluate(function () {
        console.log('window.screen', window.screen)
    })

    console.log('login');

    await mainTab.evaluate(function (config) {
        $('#numeroExterne').val(config.user_id);

        $('button.session-btn--submit').click();

        $('#numclient').val(config.user_id);
        $('#day.form-input-day').val(config.user_dob_day);
        $('#month.form-input-month').val(config.user_dob_month);
        $('#year.form-input-year').val(config.user_dob_year);
        $('.button.button--gridnum').click();
        $('#code.form-input-code').val(config.user_password);
    }, config)

    await mainTab.wait(1000);

    await mainTab.evaluate(function () {
        $('input.button--confirm').click();
    });

    await mainTab.wait(1000);

    const loginPopup = await mainTab.evaluate(function () {
        return $('.btn.cm-confirm').is(':visible')
    })

    if (loginPopup) {
        await mainTab.evaluate(function () {
            $('.btn.cm-confirm').click();
        })
        await mainTab.wait(1000);
    }

    await mainTab.evaluate(function (num, bet, typebet, simulation) {
        $('a.E_SIMPLE')[0].click();

        console.log('num', num, 'bet', bet);

        $('#participant-check-' + num)[0].click();
    }, num, bet, typebet, simulation)

    await mainTab.wait(1000);

    await mainTab.evaluate(function (num, bet, typebet, simulation) {
        console.log($('#pari-variant-GAGNANT').attr('class'))

        if (typebet == 'gagnant')
            $('#pari-variant-GAGNANT')[0].click()
        else
            $('#pari-variant-PLACE')[0].click();

        var b = Math.round(bet / 1.5);
        $('.pari-mise-input').val(b);
        $('.pari-mise-input').change();

        if (!simulation) {
            console.log('pari total', $('.pari-total').text());
            console.log('bet validate button exists', $('#pari-validate').attr('class'));
            $('#pari-validate').click();
        }
    }, num, bet, typebet, simulation)


    await mainTab.wait(1000);

    await mainTab.resizeFullScreen()
    await mainTab.saveScreenshot('/tmp/cap1')

    await browser.close()

    process.exit(0);
}

if (process.argv.length != 7) {
    console.log('Usage: bet.js <URL> <num> <bet> <type> <simulation>');
    process.exit(1);
} else {

    address = process.argv[2];
    num = process.argv[3];
    bet = process.argv[4];
    typebet = process.argv[5];
    simulation = process.argv[6] == '0' ? false : true;

    Promise.race([
        play(),
        new Promise(function (_, reject) { setTimeout(function () { reject(new Error('timeout')) }, 60000) })
    ]).catch(function (err) {
        console.log(err);
        browser.close();
        process.exit(1);
    })
}
