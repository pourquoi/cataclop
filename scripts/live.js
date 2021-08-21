const
    puppeteer = require('puppeteer')
    process = require('process')
    fs = require('fs'),
    http = require('http')
    https = require('https')
    axios = require('axios')
    ;

let programme;

const date = new Date();
const date_f = date.getDate().toString().padStart(2, '0') +
    (date.getMonth()+1).toString().padStart(2, '0') +
    date.getFullYear().toString()
    ;

function filterReunion(reunion) {
    //if (reunion.numOfficiel>1) return false;
    return true;
}

function filterRace(race) {
    //if (race.discipline != 'PLAT') return false;
    //if (race.categorieParticularite != 'HANDICAP') return false;
    if (race.categorieStatut != 'A_PARTIR') return false;
    if (race.departImminent) return false;
    //if (race.heureDepart < date.getTime()) return false;

    return true;
}

function getProgramme() {
    let promise = new Promise( (resolve, reject) => {
        let uri = '/rest/client/1/programme/';
        uri += date_f;
        uri += '?specialisation=OFFLINE&meteo=true';

        const req = https.request('https://online.turfinfo.api.pmu.fr' + uri, (res) => {
            let data = '';

            res.on('data', d => {
                data += d;
            });

            res.on('end', () => {
                resolve(JSON.parse(data));
            })
        });
        req.on('error', err => {
            console.error(err);
            reject(err);
        });
        req.end();
    })

    return promise;
}

const config = JSON.parse(fs.readFileSync('./scripts/live_config.json', 'utf8'));
axios.defaults.headers['Content-Type'] = 'application/json';

async function login() {
    console.log('login');
    try {
        const r = await axios.post(config.api_url + '/token/', {
            email: config.api_email,
            password: config.api_password
        })
        console.log(r.data);
        axios.defaults.headers.common['Authorization'] = 'Bearer ' + r.data.token;
        return true;
    } catch(err) {
        console.error(err);
        return false;
    }
}

async function play() {
    let auth = await login();
    if (!auth) return;

    programme = await getProgramme();

    console.log(programme);

    const browser = await puppeteer.launch({
        headless: false,
        args: [`--window-size=1366,768`]
    });

    setTimeout(async () => {
        const now = new Date();
        if (now.getDate() != date.getDate()) {
            console.log('daily live end');
            browser.close();
            process.exit(0);
        }
    }, 30000);

    console.log('find races');

    for(let reunion_i=0; reunion_i<programme.programme.reunions.length; reunion_i++) {
        const reunion = programme.programme.reunions[reunion_i];

        if (!filterReunion(reunion)) continue;

        for (let race_i=0; race_i<reunion.courses.length; race_i++) {
            const race = reunion.courses[race_i];

            if (!filterRace(race)) continue;

            const page = await browser.newPage();

            const race_key = 'R' + reunion.numOfficiel + '/C' + race.numOrdre;

            const address = 'https://info.pmu.fr/programme/courses/' + 
                date_f + '/' + race_key;

            page.on('request', interceptedRequest => {
                interceptedRequest.continue();
            });

            page.on('response', r => {
                //console.log(r);
                const url = r.request().url();

                if (url.includes('/rapports/SIMPLE_GAGNANT')) {
                    console.log(new Date(), race_key);

                    if (r.status() == 204) return;

                    if (race.heureDepart < date.getTime()) {
                        page.close();
                        return;
                    }

                    r.json().then(json => {
                        //console.log(json);

                        let payload = {
                            date: date.getFullYear() + '-' + (date.getMonth()+1).toString().padStart(2, '0') + '-' + date.getDate().toString().padStart(2, '0'),
                            R: reunion.numOfficiel,
                            C: race.numOrdre,
                            odds: json.rapportsParticipant.map(odds => {
                                return {
                                    player: odds.numerosParticipant[0],
                                    value: odds.rapportDirect,
                                    ts: json.dateMajDirect / 1000,
                                    offline: true,
                                    evolution: odds.tendance,
                                    whale: odds.grossePrise
                                };
                            })
                        };

                        axios.post(config.api_url + '/live_odds', payload).then(r => {

                        })
                        .catch(err => {
                            console.error('failed saving live odds');
                        })
                        ;

                        //console.log(evolution);
                    }).catch(err => console.error('bad json'))
                }
            });

            console.log(address);

            await page.setRequestInterception(true);
            await page.goto(address);
        }
    }    
}

play();