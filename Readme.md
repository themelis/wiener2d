# Wiener filtering for mass mapping 

to compile use this command, 

```
cd src/
g++ -o wiener2d wiener2d.cpp -std=c++11 -I/usr/local/include -L/usr/local/lib -lcfitsio -lfftw3 -lm
```

then, to execute the binary

```
rm wiener.fits
./wiener2d
```

and display results using the notebook inside ntbk/ folder

 
