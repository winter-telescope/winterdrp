import logging
import requests
import pandas as pd
import os, gzip, io

import base64
import astropy
from datetime import datetime
from astropy.time import Time
from astropy.io import ascii
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt


from winterdrp.processors.base_processor import BaseDataframeProcessor
from winterdrp.paths import get_output_dir

logger = logging.getLogger(__name__)

class SendToFritz(BaseDataframeProcessor):
    def __init__(self, 
                output_sub_dir: str, 
                # token = '8f59cf92-c43a-4206-a6fe-64913da58bf6', #winter2
                token = '18162129-49f8-4b85-9e0e-38f18fac32c1',
                group_ids = [1431],
                base_name = 'WIRC',
                *args,
                **kwargs):
        super(SendToFritz, self).__init__(*args, **kwargs)
        self.token = token
        self.group_ids = group_ids
        self.base_name = base_name

    def _apply_to_candidates(
            self,
            candidate_table: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("In SendToFritz")
        self.make_alert(candidate_table) 
        return candidate_table

    def open_bytes_obj(self, bytes_obj):
        """Return numpy array of bytes_obj
        
        Args:
            bytes_obj (_io.BytesIO object in memory): BytesIO obj representing image.

        Returns:
            numpy.ndarrary: representation of the image        
        """
        bytes_io = io.BytesIO(gzip.open(io.BytesIO(bytes_obj.getvalue())).read()) # io.BytesIO obj, ready to be read by fits.open    
        cutout = fits.open(bytes_io)[0].data
        return cutout

    def read_input_df(self, df):
        """Takes a DataFrame, which has multiple candidate 
        and creates list of dictionaries, each dictionary 
        representing a single candidate.

        NOTE: saves images as np.arrays for thumbnail sending.

        Args:
            df (pandas.core.frame.DataFrame): dataframe of all candidates.
        
        Returns:
            (list[dict]): list of dictionaries, each a candidate.
        """
        all_candidates = []   
        
        for i in range(0, len(df)):
            candidate = {} 
            for key in df.keys():
                try: 
                    if type(df.iloc[i].get(key)) is str:
                        candidate[key] = df.iloc[i].get(key)   
                    else:
                        # change to native python type
                        candidate[key] = df.iloc[i].get(key).item()
                except AttributeError: # for IOBytes objs
                    candidate[key] = self.open_bytes_obj(df.iloc[i].get(key))
                                                 
            all_candidates.append(candidate)

        return all_candidates 

    def get_next_name(self, lastname, candjd, bwfile = 'badwords.txt', begcount = 'aaaaaaa'):
        """Creates candidate name following the naming format of 'WNTR22aaaaaaa' .
        Modified from https://github.com/dekishalay/pgirdps

        Args:
            lastname(str): last used candidate name.
            candjd (str): candidate's JD.
            bwfile (str): file name of .txt file of excluded words. 
            begcount (str): string to start naming convention.
        
        Returns:
            (str): next candidate name in sequence.
        """ 
        curyear = Time(candjd, format = 'jd').datetime.strftime('%Y')[2:4]

        if lastname is None:
            #If this is the first source being named
            newname = self.base_name + curyear + begcount
            return newname
        
        lastyear = lastname[4:6]

        if curyear != lastyear:
            #If this is the first candidate of the new year, start with aaaaa
            newname = self.base_name + curyear + begcount
            return newname
        else:
            lastcount = lastname[6:]
            charpos = len(lastcount) - 1
            # will iteratively try to increment characters starting from the last
            inctrue = False
            usestring = ''
            while charpos >= 0:
                cref = lastcount[charpos]
                if inctrue:
                    usestring = cref + usestring
                    charpos -= 1
                    continue
                creford = ord(cref)
                #increment each character, if at 'z', increment the next one
                if creford + 1 > 122:
                    usestring = 'a' + usestring
                else:
                    nextchar = chr(creford+1)
                    usestring = nextchar + usestring
                    inctrue = True
                charpos -= 1
                continue
            
            newname = self.base_name + curyear + usestring

            curdir = os.path.dirname(__file__) # /data/sulekha/winterdrp/winterdrp/processors/alert_packets
            file_path = os.path.join(curdir, bwfile)

            bwlist = ascii.read(file_path, format = 'no_header')
            isbw = False
            # check for bad word
            for i in range(len(bwlist)):
                if usestring.find(str(bwlist['col1'][i])) != -1:
                    #Shame shame
                    isbw = True
                    break
            if isbw:
                # increment the name with a recursive call
                return self.get_next_name(newname, candjd)
            else:
                return newname

    def api(self, method, endpoint, data=None):
        headers = {'Authorization': f'token {self.token}'}
        response = requests.request(method, endpoint, json=data, headers=headers)
        return response

    def add_new_source(self, cand):
        data = {'ra': cand['ra'],
                        'dec': cand['dec'],
                        'id': cand['objectId'],
                        'group_ids': self.group_ids
        }   
        logger.info(f"dict:{data}")
        response = self.api('POST', 'https://fritz.science/api/sources', data)
        return response
    
    def upload_thumbnail(self, cand):
        """Post new thumbnail to Fritz.

        Format of thumbnail payload:
        { "obj_id": "string",  "data": "string",  "ttype": "string"}
        """
        fritz_to_cand = {"new": 'SciBitIm', "ref": 'RefBitIm', "sub": 'DiffBitIm'}

        for fritz_key in fritz_to_cand.keys():
            cand_key = fritz_to_cand[fritz_key]
            cutout = cand[cand_key]

            buffer = io.BytesIO()
            plt.figure(figsize=(3,3))
            mean, median, std = sigma_clipped_stats(cutout)
            plt.imshow(cutout, origin='lower', cmap='gray',vmin=mean-1*std,vmax=median+3*std)
            plt.xticks([])
            plt.yticks([])

            plt.savefig(buffer,format='png')

            cutoutb64 = base64.b64encode(buffer.getvalue())
            cutoutb64_string = cutoutb64.decode('utf8')

            data_payload = {'obj_id':cand['objectId'],
                            'data':cutoutb64_string,
                            'ttype':fritz_key
                        }

            response = self.api('POST', 'https://fritz.science/api/thumbnail', data=data_payload)
            # logger.info(f'candid {data_payload["obj_id"]}: {data_payload["ttype"]}, thumbnail response:{response}')
        return response

    def update_photometry(self, cand):
        """Send photometry to Fritz."""
        data_payload = {"filter": "cspjs",
                        "magsys": "vega",
                        "obj_id": cand["objectId"],
                        "mag": cand["magpsf"],
                        "magerr": cand["sigmapsf"],
                        "instrument_id": 5,
                        "mjd": cand['jd'] - 2400000.5,
                        "limiting_mag": 99,
                        "group_ids": self.group_ids
                        }
        # response = self.api('PATCH', 'https://fritz.science/api/photometry/photometry_id', data=data_payload)
        response = self.api('PUT', 'https://fritz.science/api/photometry', data=data_payload)

        # logger.info(f'candid {data_payload["obj_id"]} photo response:{response.text}')
        return response

    def create_new_cand(self, cand, id):
        """Create new candidate(s) (one per filter)"""
        data = { "id": cand["objectId"],
                    "filter_ids": [1147],
                    "passing_alert_id": 1147,
                    "passed_at": Time(datetime.utcnow()).isot,
                    "ra": cand["ra"],
                    "dec": cand["dec"],
                    }
        response = self.api('POST','https://fritz.science/api/candidates',data=data)        
        # response = self.api('POST', 'https://fritz.science/#tag/candidates/paths/~1api~1candidates/get/api/candidates',data=data)        
        logger.info(f'new create response {response.text}')
        return response


    def make_alert(self, cand_table):
        all_cands = self.read_input_df(cand_table)

        last_name = 'WIRC21aaaaaao'
        cand_id = 700
              
        for cand in all_cands:
            cand_jd = cand['jd'] # float
            cand_name = self.get_next_name(last_name, str(cand_jd))

            #TODO candid should be coming from naming database
            cand['candid'] = cand_id
            cand_id += 1
            
            # TODO check if cand is new? 
            # cand_name, new_status = self.check_and_insert_source(cand_name, cand)

            #TODO remove once new_status is up-to-date: dummy line
            new_status = True

            if new_status:
                last_name = cand_name    
            cand['objectId'] = cand_name

            # source_response = self.add_new_source(cand)
            source_response = self.create_new_cand(cand, id)
            logger.info(f'Add source {cand["objectId"]}: {source_response}')
            thumbnail_response = self.upload_thumbnail(cand)
            logger.info(f'Upload thumbnail {cand["objectId"]}: {thumbnail_response}')
            photometry_response = self.update_photometry(cand)
            logger.info(f'Photometry {cand["objectId"]}: {photometry_response}')

            