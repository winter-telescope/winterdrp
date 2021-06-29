from astropy.io import fits, ascii
import numpy as np
from glob import glob

def fix_headers(imgname,logfile='log.csv'):
	img = fits.open(imgname,'update')
	img[0].header['RA'] = img[0].header['mount_ra_j2000_hours']*15
	img[0].header['DEC'] = img[0].header['mount_dec_j2000_deg']
	img[0].header['EXPTIME'] = img[0].header['exptime_actual']
	filters = {'4':'OPEN','3':'r','1':'u'}
	img[0].header['FILTER'] = filters['%i'%(int(img[0].header['Viscam_Filter_Wheel_Position']))]
	img[0].header['BZERO'] = 0

	log = ascii.read(logfile)
	obstype = log[log['imgname']==imgname.split('/')[-1]]['obstype'][0]
	img[0].header['OBSTYPE'] = str(obstype)

	img[0].data = img[0].data*1.0
	img[0].data[2048,:] = np.nan
	img.close()



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--d",type=str,help='Dirname')
	parser.add_argument("--l",type=str,default='log.txt',help='log file')
	args = parser.parse_args()

	imglist = glob('%s/*.fits'%(args.d))
	for imgname in imglist:
		print(imgname)
		fix_headers(imgname,args.l)