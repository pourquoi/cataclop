import os
import shutil
from django.test import TestCase, tag

from cataclop.pmu.scrapper import Scrapper, PmuScrapper, UnibetScrapper


@tag('integration')
class ScrapperTest(TestCase):
    def setUp(self):
        if os.path.exists('/tmp/scrap'):
            shutil.rmtree('/tmp/scrap')
        os.mkdir('/tmp/scrap', 0o777)

    def tearDown(self):
        shutil.rmtree('/tmp/scrap')
        pass

    def test_parse(self):
        scrapper = Scrapper('/tmp/scrap')
        scrapper.scrap()

        dirs = os.listdir('/tmp/scrap')
        self.assertEqual(1, len(dirs))
        self.assertTrue(os.path.exists('/tmp/scrap/' + dirs[0] + '/programme.json'))
