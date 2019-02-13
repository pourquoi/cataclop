const
    HeadlessChrome = require('simple-headless-chrome')
process = require('process')
fs = require('fs')
    ;


var config = JSON.parse(fs.readFileSync('./scripts/bet_config.json', 'utf8'));

const browser = new HeadlessChrome({
    headless: !!config.headless, // If you turn this off, you can actually see the browser navigate with your instructions
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


var address, bet, bets, num, typebet, simulation = false;

var debug = 1;

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

    console.log('login');

    await mainTab.evaluate(function (config) {

        $('.login-input[name="username"]')[0].value = config.unibet.user_id
        $('.login-input[name="password"]')[0].value = config.unibet.user_password
        $('.form-dob[name="form-dob"]')[0].value = config.unibet.user_dob_day + ' / ' + config.unibet.user_dob_month + ' / ' + config.pmu.user_dob_year;

        $('input.button-login[type="submit"]')[0].click()
        
    }, config)

    await mainTab.wait(3000);

    const loginPopup = await mainTab.evaluate(function () {
        return $('#modal-arjel-confirm').is(':visible')
    })

    if (loginPopup) {
        await mainTab.evaluate(function () {
            $('#modal-arjel-confirm')[0].click()
        })
        await mainTab.wait(2000);
    }

    await mainTab.evaluate(function() {
        $('a[data-turf-category="SIMPLE"]')[0].click()
    })

    var b_idx = 0;
    for(;b_idx<bets.length;b_idx++) {

        num = bets[b_idx][0]
        bet = bets[b_idx][1]

        await mainTab.evaluate(function(num) {
            $('li[data-runner-rank="' + num + '"] i.bet[data-turf-bettype-id="1"]')[0].click()
        }, num)

    }

    await mainTab.evaluate(function() {
        $('a#turf_betslip_place')[0].click()
    })

    await mainTab.wait(1000);

    await mainTab.evaluate(function(simulation) {
        console.log('bet validate button exists', $('#turf_betslip_confirm').attr('class'));

        if( !simulation ) {
            console.log('betting');
            $('#turf_betslip_confirm')[0].click()
        }
    }, simulation)

    await mainTab.wait(1000);
    
    //await browser.close()

    //process.exit(0);
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
        browser.close();
        process.exit(1);
    })
}



