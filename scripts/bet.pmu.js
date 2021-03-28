const
    puppeteer = require('puppeteer')
    process = require('process')
    fs = require('fs')
    ;

var config = JSON.parse(fs.readFileSync('./scripts/bet_config.json', 'utf8'));

var address, bet, bets, num, typebet, simulation = false;

var debug = 1;

async function play() {
    console.log('init browser');

    const browser = await puppeteer.launch({
        headless: !!config.headless,
        args: [`--window-size=1366,768`]
    });

    console.log('open new tab');

    const page = await browser.newPage();
    
    page.on('console', function (msg) {
        console.log(msg)
    })

    console.log('open address');

    await page.goto(address);
    await page.waitFor(2000);

    await page.addScriptTag({url: 'https://code.jquery.com/jquery-3.2.1.min.js'})

    page.evaluate(function () {
        console.log('window.screen', window.screen)
    })

    console.log('login');

    await page.evaluate(function (config) {
        $('#numeroExterne').val(config.pmu.user_id);

        $('button.session-btn--submit').click();

        $('#numclient').val(config.pmu.user_id);
        $('#day.form-input-day').val(config.pmu.user_dob_day);
        $('#month.form-input-month').val(config.pmu.user_dob_month);
        $('#year.form-input-year').val(config.pmu.user_dob_year);
        $('.button.button--gridnum').click();
        $('#code.form-input-code').val(config.pmu.user_password);
    }, config)

    await page.waitFor(1000);

    await page.evaluate(function () {
        $('input.button--confirm').click();
    });

    await page.waitFor(1000);

    const loginPopup = await page.evaluate(function () {
        return $('.btn.cm-confirm').is(':visible')
    })

    if (loginPopup) {
        await page.evaluate(function () {
            $('.btn.cm-confirm').click();
        })
        await page.waitFor(3000);
    }

    var b_idx = 0;
    for(;b_idx<bets.length;b_idx++) {

        num = bets[b_idx][0]
        bet = bets[b_idx][1]

        await page.evaluate(function (num, bet, typebet, simulation) {
            $('a.E_SIMPLE')[0].click();

            console.log('num', num, 'bet', bet);

            $('#participant-check-' + num)[0].click();
        }, num, bet, typebet, simulation)

        await page.waitFor(1000);

        await page.evaluate(function (num, bet, typebet, simulation) {
            console.log($('#pari-variant-GAGNANT').attr('class'))

            if (typebet == 'gagnant')
                $('#pari-variant-GAGNANT')[0].click()
            else
                $('#pari-variant-PLACE')[0].click();

            var b = Math.round(bet / 1.5);
            $('.pari-mise-input').val(b);
            $('.pari-mise-input').change();

            $('.btn-mise-increment').click();

            console.log('pari total', $('.pari-total').text());
            console.log('bet validate button exists', $('#pari-validate').attr('class'));
            
            if (!simulation) {
                $('#pari-validate').click();
            }
        }, num, bet, typebet, simulation)

        if( b_idx != bets.length-1 )
            await page.waitFor(3000);
        else 
            await page.waitFor(1000);
    }

    await browser.close()

    process.exit(0);
}

if (process.argv.length != 6) {
    console.log("Usage: bet.js <URL> <bets> <type> <simulation>\n");
    process.exit(1);
} else {

    address = process.argv[2];
    bets_cmd = process.argv[3];
    typebet = process.argv[4];
    simulation = process.argv[5] == '0' ? false : true;

    bets = bets_cmd.split(',').map( b => {
        return b.split(':')
    })

    Promise.race([
        play(),
        new Promise(function (_, reject) { setTimeout(function () { reject(new Error('timeout')) }, 60000) })
    ]).catch(function (err) {
        console.log(err);
        process.exit(1);
    })
}
