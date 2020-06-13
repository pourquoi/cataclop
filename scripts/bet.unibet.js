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

    console.log('login');

    await page.evaluate(function (config) {

        $('.login-input[name="username"]')[0].value = config.unibet.user_id
        $('.login-input[name="password"]')[0].value = config.unibet.user_password
        $('.form-dob[name="form-dob"]')[0].value = config.unibet.user_dob_day + ' / ' + config.unibet.user_dob_month + ' / ' + config.pmu.user_dob_year;

        $('input.button-login[type="submit"]')[0].click()
        
    }, config)

    await page.waitFor(3000);

    console.log('logout')

    await page.evaluate(function() {
        $('.ui-action-logout')[0].click()
    })

    await page.waitFor(2000);


    console.log('login');

    await page.evaluate(function (config) {

        $('.login-input[name="username"]')[0].value = config.unibet.user_id
        $('.login-input[name="password"]')[0].value = config.unibet.user_password
        $('.form-dob[name="form-dob"]')[0].value = config.unibet.user_dob_day + ' / ' + config.unibet.user_dob_month + ' / ' + config.pmu.user_dob_year;

        $('input.button-login[type="submit"]')[0].click()
        
    }, config)

    await page.waitFor(3000);


    const loginPopup = await page.evaluate(function () {
        return $('#modal-arjel-confirm').is(':visible')
    })

    console.log(loginPopup)

    if (loginPopup) {
        await page.evaluate(function () {
            if($('#arjel_terms_checkbox').length) {
                $('#arjel_terms_checkbox')[0].click()
            }
            $('#modal-arjel-confirm')[0].click()
        })
        await page.waitFor(2000);
    }

    await page.evaluate(function() {
        $('a[data-turf-category="SIMPLE"]')[0].click()
    })

    await page.waitFor(3000);

    var b_idx = 0;
    for(;b_idx<bets.length;b_idx++) {

        num = bets[b_idx][0]
        bet = bets[b_idx][1]

        await page.evaluate(function(num) {
            $('li[data-runner-rank="' + num + '"] i.bet[data-turf-bettype-id="1"]')[0].click()
        }, num)

    }

    await page.waitFor(3000);

    await page.evaluate(function() {
        $('a#turf_betslip_place')[0].click()
    })

    await page.waitFor(30000);

    await page.evaluate(function(simulation) {
        console.log('bet validate button exists', $('#turf_betslip_confirm').attr('class'));

        if( !simulation ) {
            console.log('betting');
            $('#turf_betslip_confirm')[0].click()
        }
    }, simulation)

    await page.waitFor(1000);
    
    await browser.close()

    process.exit(0);
}

if (process.argv.length != 6) {
    console.log("Usage: bet.unibet.js <URL> <bets> <type> <simulation>\n");
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



