
#include <string.h>
#include <fftw3.h>
#include <iostream>
#include <complex>
#include <vector>
#include <numeric>
#include <algorithm>

#include "fitsio.h"

using namespace std;

std::vector<double> powerspec2d(std::vector<double> spectrum, int npix, int impix)
{
	// this function will work for square images only
	std::vector<double> ps2d(impix, 0.0);

	long unsigned int nspec = spectrum.size();

	int *ni = (int *) malloc(sizeof(int) * npix);
	for (int i = 0; i < npix; i++)
		ni[i] = int(i - npix / 2.);

	int dr;
	int ind = 0;
	for (int i=0; i < npix; i++){
		for (int j=0; j < npix; j++){
			dr = int(sqrt(ni[i] * ni[i] + ni[j] * ni[j]));
			if ( (dr == 0) || (dr>nspec-1))
				ps2d[ind] = 1e-19;
			else
				ps2d[ind] = spectrum[dr];
			++ind;
		}
	}

	return ps2d;
}

void fftshift(complex<double> *image, complex<double> *shiftedimg, int npix)
{

  complex<double> array2d[npix][npix];
  complex<double> rdr2d[npix][npix];

  // initialize
  int ind1 = 0;
  for (int x = 0; x < npix; x++) {
    for (int y = 0; y < npix ; y++) {
        array2d[x][y] = image[ind1];
        rdr2d[x][y] = {0.0,0.0};
        ++ind1;
		}
	}

	// reorder
	for (int x = 0; x < npix /2 ; x++){
        for (int y = 0; y < npix / 2; y++)  {
            rdr2d[x + npix / 2][y + npix / 2] = array2d[x][y];
		}
	}

	for (int x = 0; x < npix /2 ; x++){
        for (int y = npix/2; y < npix; y++)  {
            rdr2d[x + npix / 2][y - npix / 2] = array2d[x][y];
		}
	}

	for (int x = npix /2; x < npix ; x++){
        for (int y = npix/2; y < npix; y++)  {
            rdr2d[x - npix / 2][y - npix / 2] = array2d[x][y];
		}
	}

	for (int x = npix /2; x < npix ; x++){
        for (int y = 0; y < npix /2; y++)  {
            rdr2d[x - npix / 2][y + npix / 2] = array2d[x][y];
		}
	}
	// store
	int ind2 = 0;
    for (int x = 0; x < npix; x++) {
        for (int y = 0; y < npix ; y++) {
            shiftedimg[ind2] = rdr2d[x][y];
            ++ind2;
		}
	}

}


std::vector<std::complex<double>> H_operator(std::vector<std::complex<double>> kappa, int npix)
{
	int impix = npix * npix;
 	double fftFactor = 1.0 / ((double) impix);
	fftw_complex *fft_frameHf0 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * impix);
	fftw_complex *fft_frameHb0 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * impix);
	fftw_plan planHf0 = fftw_plan_dft_2d(npix, npix, fft_frameHf0, fft_frameHf0, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan planHb0 = fftw_plan_dft_2d(npix, npix, fft_frameHb0, fft_frameHb0, FFTW_BACKWARD, FFTW_MEASURE);

	std::vector<std::complex<double>> result(impix, {.0, .0});
	double freqFactor = 1. / ((double) npix); // 2.0 * M_PI / ((double) npix ) ; // 1.0 / ((double) npix); // / pixel_size

	for (long ind = 0; ind < impix; ind++) {
			fft_frameHf0[ind][0] = kappa[ind].real();
			fft_frameHf0[ind][1] = kappa[ind].imag();
	}

	fftw_execute(planHf0);

  double k1, k2, k1k1, k2k2, k1k2, ksqr;
	double c1, c2;

  for (int y = 0; y < npix ; y++) {
		k2  = (y < npix / 2 ? y * freqFactor : (y - npix) * freqFactor); // this comes from python fft

		for (int x = 0; x < npix ; x++) {
      k1  = (x < npix / 2 ? x * freqFactor : (x - npix) * freqFactor); // this comes from python fft

			long pos = y * npix + x ;
      if (k1 == 0 && k2 == 0) {
          fft_frameHb0[pos][0] = 0.;
          fft_frameHb0[pos][0] = 0.;
          continue;
      }

			k1k1 = k1 * k1;
			k2k2 = k2 * k2;
			ksqr = k1k1 + k2k2;
			c1 = k1k1 - k2k2;
			c2 = 2.0 * k1 * k2;

      fft_frameHb0[pos][0] = ( fft_frameHf0[pos][0] * c1 + fft_frameHf0[pos][1] * c2)  / ksqr * fftFactor;
		  fft_frameHb0[pos][1] = (-fft_frameHf0[pos][0] * c2 + fft_frameHf0[pos][1] * c1)  / ksqr * fftFactor;

		}
	}

	fftw_execute(planHb0);

	// store computed kappa to output
	for (long ind = 0; ind < impix; ind++)
		result[ind] = {fft_frameHb0[ind][0], fft_frameHb0[ind][1]};

	return result;

}

std::vector<std::complex<double>> H_adjoint_operator(std::vector<std::complex<double>> gamma, int npix)
{

	int impix = npix * npix;
	double fftFactor = 1.0 / ((double) impix);
	fftw_complex *fft_frameHf0 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * impix);
	fftw_complex *fft_frameHb0 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * impix);
	fftw_plan planHf0 = fftw_plan_dft_2d(npix, npix, fft_frameHf0, fft_frameHf0, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan planHb0 = fftw_plan_dft_2d(npix, npix, fft_frameHb0, fft_frameHb0, FFTW_BACKWARD, FFTW_MEASURE);

	std::vector<std::complex<double>> result(impix, {.0, .0});
	double freqFactor = 1. / ((double) npix);

  // apply lensing to shear values here
  for (long i = 0; i < impix ; i++) {
    fft_frameHf0[i][0] = gamma[i].real();
    fft_frameHf0[i][1] = gamma[i].imag();
  }

  fftw_execute(planHf0);

	double k1, k2, k1k1, k2k2, k1k2, ksqr;
	double c1, c2;

	for (int y = 0; y < npix ; y++) {
    k2  = (y < npix / 2  ? y * freqFactor : (y - npix) * freqFactor);

		for (int x = 0; x < npix ; x++) {
			k1 = (x < npix / 2 ? x * freqFactor : (x - npix) * freqFactor);

      long pos = y * npix + x;

      if (k1 == 0 && k2 == 0){
        fft_frameHb0[pos][0] = 0.;
        fft_frameHb0[pos][1] = 0.;
        continue;
      }

			k1k1 = k1 * k1;
			k2k2 = k2 * k2;
			ksqr = k1k1 + k2k2;
			c1 = k1k1 - k2k2;
			c2 = 2.0 * k1 * k2;

      fft_frameHb0[pos][0] = ((fft_frameHf0[pos][0] * c1 - fft_frameHf0[pos][1] * c2)  / ksqr) * fftFactor ;
		  fft_frameHb0[pos][1] = ((fft_frameHf0[pos][0] * c2 + fft_frameHf0[pos][1] * c1)  / ksqr) * fftFactor ;

		}
	}

	fftw_execute(planHb0);

	// store computed kappa to output
  for (long ind = 0; ind < impix; ind++)
    result[ind] = {fft_frameHb0[ind][0], fft_frameHb0[ind][1]};

  return result;
}

double* wiener_core(std::vector<std::complex<double>> shear_res, std::vector<double> Px, std::vector<double> Sn, unsigned int Niter, int npix)
{

	int impix = npix * npix;
	double fftFactor = 1.0 / ((double) impix);

	// alocate result
	double * result = (double *) malloc(impix * sizeof(double));

	// allocate fftw plans
	fftw_complex *frame = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * impix);
	fftw_plan planf = fftw_plan_dft_2d(npix, npix, frame, frame, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan planb = fftw_plan_dft_2d(npix, npix, frame, frame, FFTW_BACKWARD, FFTW_MEASURE);

	// initiallize
	std::vector<std::complex <double>> t(impix, {.0,.0});
	std::vector<std::complex <double>> xg(impix, {.0,.0});

	// find the minimum noise variance
	std::vector<double>::iterator itr = std::min_element(Sn.begin(), Sn.end());
	int index  = std::distance(Sn.begin(), itr);
	double tau = Sn[index];

	// set the step size
	double eta = 1.83 * tau;

	std::vector<double> Eta(impix, eta);
	std::vector<double> Wfc(impix, .0);
	std::vector<double> Esn(impix, .0); //(std::complex<double>){.0, .0}
	std::transform(Eta.begin(), Eta.end(), Sn.begin(), Esn.begin(), std::divides<double>()); // eta / Sn

	// calculate the wiener filter coefficients
	std::transform(Px.begin(), Px.end(), Eta.begin(), Wfc.begin(), std::plus<double>()); // Wfc = Px + Pn
	std::transform(Px.begin(), Px.end(), Wfc.begin(), Wfc.begin(), std::divides<double>()); // Wfc = Px / (Px + Pn)

	// main loop
	for (int iter = 0; iter < Niter; ++iter)
	{
		// print alive
		if (iter % 50 == 0) {
				std::cout << "Wiener iteration :" << iter << std::endl;
		}

		// t equation
		t = H_operator(xg, npix); // H * xg
		std::transform(shear_res.begin(), shear_res.end(), t.begin(), t.begin(), std::minus<std::complex<double>>()); // y - H * xg
		std::transform(Esn.begin(), Esn.end(), t.begin(), t.begin(), std::multiplies<std::complex<double>>()); // (eta / Sn) * (y - H * xg)

		t = H_adjoint_operator(t, npix); // H^T * ( (eta / Sn) * (y-H*xg) )

		std::transform(t.begin(), t.end(), xg.begin(), t.begin(), std::plus<std::complex<double>>()); // t = xg + H^T * ( (eta / Sn) * (y - H * xg) )

		// compute the fft of t
		for (int ind =0; ind < impix; ++ind)
		{
			frame[ind][0] = t[ind].real();
			frame[ind][1] = t[ind].imag();
		}
		fftw_execute(planf);
		for (int ind = 0; ind < impix ; ind++)
			t[ind] = {frame[ind][0], frame[ind][1]};

		// apply filter in fourier space
		std::transform(Wfc.begin(), Wfc.end(), t.begin(), xg.begin(), std::multiplies<std::complex<double>>());

		// compute the inverse fft of xg
		for (int ind =0; ind < impix; ++ind)
		{
			frame[ind][0] = xg[ind].real() * fftFactor;
			frame[ind][1] = xg[ind].imag() * fftFactor;
		}
		fftw_execute(planb);
		for (int ind = 0; ind < impix ; ind++) {
			xg[ind] = {frame[ind][0], frame[ind][1]};
    }

	}

	for (int ind = 0; ind < impix ; ind++)
		result[ind] = (double) (real(xg[ind]));

	return result;
}



int main(int argc, char** argv)
{

	// ----------------------------------------------------------------
	// read fixed input fits files
	int npix  = 256;
	int impix = 256 * 256;
	int pslen = 182;

	double *dat0 = (double *) malloc( impix * sizeof(double));
	double *dat1 = (double *) malloc( impix * sizeof(double));
	double *dat2 = (double *) malloc( impix * sizeof(double));
	double *dat3 = (double *) malloc( pslen * sizeof(double));

 	fitsfile *fptr;
	int status = 0;
	long fpixel[2] = {1,1};
	int *anynull = 0;
	int *nullval = 0;
	fits_open_image(&fptr, "../data/mice_g1_map.fits", READONLY, &status);
	fits_read_pix(fptr, TDOUBLE, fpixel, (long) (impix), anynull, dat0, nullval, &status);
	fits_close_file(fptr, &status);
	fits_open_image(&fptr, "../data/mice_g2_map.fits", READONLY, &status);
	fits_read_pix(fptr, TDOUBLE, fpixel, (long) (impix), anynull, dat1, nullval, &status);
	fits_close_file(fptr, &status);
	fits_open_image(&fptr, "../data/mice_noisecov_map.fits", READONLY, &status);
	fits_read_pix(fptr, TDOUBLE, fpixel, (long) (impix), anynull, dat2, nullval, &status);
	fits_close_file(fptr, &status);
	fits_open_image(&fptr, "../data/mice_cosmosis_ps1d_kappa.fits", READONLY, &status);
	fits_read_pix(fptr, TDOUBLE, fpixel, (long) (pslen), anynull, dat3, nullval, &status);
	fits_close_file(fptr, &status);

	// ----------------------------------------------------------------


	// pushback data to vectors
	std::vector<double> ps1d;
	ps1d.reserve(pslen);
	for (int i=0; i < pslen; i++)
		ps1d.push_back(dat3[i]);

	// constract 2d square powermap out of the 1d power spectrum
	std::vector<double> ps2d = powerspec2d(ps1d, npix, impix);
	// fftshift the resulting 2d power spectrum
	std::complex<double> * rimg = (std::complex<double> *) malloc(sizeof(std::complex<double>) * impix);
	std::complex<double> * shimg = (std::complex<double> *) malloc(sizeof(std::complex<double>) * impix);
	for (int i = 0; i < impix ; i++)
		rimg[i] = {ps2d[i], .0};
	fftshift(rimg, shimg, npix);

	// pushback data to vectors
	std::vector<double> Sn;
	std::vector<double> Px;
	std::vector<std::complex<double>> shear_res;
	Sn.reserve(impix);
	Px.reserve(impix);
	shear_res.reserve(impix);
	for (int i=0; i < impix; i++) {
		Sn.push_back(dat2[i]);
		Px.push_back(shimg[i].real() );
		shear_res.push_back({dat0[i], -dat1[i]});
	}

	// run wiener main
	int Niter = 101;
	double * imgw = (double *) malloc(impix * sizeof(double));

	cout << "Running wiener filtering" << endl;
	imgw = wiener_core(shear_res, Px, Sn, Niter, npix);
	cout << "Wiener map computed" << endl;

	// write wiener estimate to a fits file
	long naxes[2] = { npix, npix };
	fits_create_file(&fptr, "./wiener.fits", &status);
	fits_create_img(fptr,  DOUBLE_IMG, 2, &naxes[0], &status);
	fits_write_img(fptr, TDOUBLE, 1, impix, imgw, &status);
  fits_close_file(fptr, &status);

  return 0;
}
