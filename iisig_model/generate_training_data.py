from dataclasses import dataclass
from faker import Faker
import pandas as pd
import numpy as np
from xeger import Xeger

@dataclass
class TrainingDataMaker:
    num_rows: int = 10_000
    fake = Faker('en_GB')
    x = Xeger(limit=10)

    def main(self):
        n_rows_per_regex = int(np.floor(self.num_rows / 3))

        first_names = [self.fake.first_name() for _ in range(self.num_rows)]
        last_names = [self.fake.last_name() for _ in range(self.num_rows)]
        full_name = [self.fake.name() for _ in range(self.num_rows)]
        full_address = [self.fake.address().replace('\n', '  ') for _ in range(self.num_rows)]
        street_address = [self.fake.street_address().replace('\n', '  ') for _ in range(self.num_rows)]
        country = [self.fake.country() for _ in range(self.num_rows)]

        place1 = [self.fake.city() for _ in range(n_rows_per_regex)]
        place2 = [self.fake.county() for _ in range(n_rows_per_regex)]
        place3 = [self.fake.administrative_unit() for _ in range(n_rows_per_regex)]

        regex1 = [self.x.xeger('[A-Z][a-z0-9]{5}') for _ in range(n_rows_per_regex)]
        regex2 = [self.x.xeger('[A-Z][a-z]{2}[0-9]{2}[A-Z]') for _ in range(n_rows_per_regex)]
        regex3 = [self.x.xeger('[A-Z0-9]{6}') for _ in range(n_rows_per_regex)]

        first_names = pd.DataFrame({'Value': first_names, 'Label': 'first_name'})
        last_names = pd.DataFrame({'Value': last_names, 'Label': 'last_name'})
        full_name = pd.DataFrame({'Value': full_name, 'Label': 'full_name'})
        regex1 =  pd.DataFrame({'Value': regex1, 'Label': 'regex'})
        regex2 =  pd.DataFrame({'Value': regex2, 'Label': 'regex'})
        regex3 =  pd.DataFrame({'Value': regex3, 'Label': 'regex'})
        full_address = pd.DataFrame({'Value': full_address, 'Label': 'full_address'})
        street_address = pd.DataFrame({'Value': street_address, 'Label': 'street_address'})
        country = pd.DataFrame({'Value': country, 'Label': 'country'})
        place1 = pd.DataFrame({'Value': place1, 'Label': 'place'})
        place2 = pd.DataFrame({'Value': place2, 'Label': 'place'})
        place3 = pd.DataFrame({'Value': place3, 'Label': 'place'})
        df = pd.concat([first_names, last_names, full_name, regex1, regex2, regex3, full_address,
                        street_address, country, place1, place2, place3])
        df = df.sample(frac=1)
        df = df.reset_index(drop=True)

        return df