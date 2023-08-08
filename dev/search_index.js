var documenterSearchIndex = {"docs":
[{"location":"derivations/#Derivations","page":"Derivations","title":"Derivations","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Here we will derive properties of correlation functions in imaginary time and their spectral functions. If you find an error in this page please report an issue via github or email James Neuhaus. ","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Conventions: hbar=1, for pmmp top signs are for bosonic correlations, bottom signs fermionic","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Causal Matsubara correlation functions are defined such that for operators hatAhatB the correlation function is ","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"C_AB(tau -tau)=-langle T_tauA(tau)B(tau)rangle=-Theta(tau-tau)langle A(tau)B(tau)rangle mp Theta(tau-tau)langle B(tau)A(tau)rangle ","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"By using translational symmetry we can set tau=0 which we will do going forward.","category":"page"},{"location":"derivations/#Periodicity","page":"Derivations","title":"Periodicity","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Starting from tau  0 we have","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nC_ABleft(tau0right)=-langle Aleft(tauright)Bleft(0right)rangle=  -frac1ZTrlefte^-beta He^tau HAe^-tau HBright\n=  -frac1ZTrleftBe^-beta He^tau HAe^-tau Hright\n=  -frac1ZTrleftBe^-beta He^tau HAe^-tau Hleft(e^beta He^-beta Hright)right\n=  -frac1ZTrlefte^-beta HBe^-beta He^tau HAe^-tau He^beta Hright\n=  -frac1ZTrlefte^-beta HBe^left(tau-betaright)HAe^-left(tau-betaright)Hright\n=  -langle Bleft(0right)Aleft(tau-betaright)rangle\n=  pmlangle T_tauAleft(tau-betaright)Bleft(0right)rangle\n=  pm C_ABleft(tau-betaright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Thus, correlation functions take on the patterns:","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"(Image: Cristophe Berthod, https://giamarchi.unige.ch/wp-content/php\\_code/people/christophe.berthod/pdf/Many-body.pdf) Cristophe Berthod","category":"page"},{"location":"derivations/#Reflection-Symmetry","page":"Derivations","title":"Reflection Symmetry","text":"","category":"section"},{"location":"derivations/#Correlation-Functions","page":"Derivations","title":"Correlation Functions","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Let us look at correlation functions with the operator order flipped","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nC_BAleft(tauright)=  -Thetaleft(tauright)leftlangle Bleft(tauright)Aleft(0right)rightrangle mpThetaleft(-tauright)leftlangle Aleft(0right)Bleft(tauright)rightrangle \n=  -fracThetaleft(tauright)ZTrlefte^-beta He^tau HBe^-tau HArightmpfracThetaleft(-tauright)ZTrlefte^-beta HAe^tau HBe^-tau Hright\n=  -fracThetaleft(tauright)ZTrlefte^-beta He^beta He^-tau HAe^-beta He^tau HBrightmpfracThetaleft(-tauright)ZTrlefte^-beta HAe^tau He^beta He^-beta HBe^-tau Hright\n=  -fracThetaleft(tauright)ZTrlefte^-beta He^beta He^-tau HAe^-beta He^tau HBrightmpfracThetaleft(-tauright)ZTrlefte^-beta HBe^-tau He^-beta HAe^tau He^beta Hright\n=  -fracThetaleft(tauright)ZTrlefte^-beta He^left(beta-tauright)HAe^-left(beta-tauright)He^tau HBrightmpfracThetaleft(-tauright)ZTrlefte^-beta HBe^-left(beta+tauright)HAe^left(beta+tauright)Hright\n=  -Thetaleft(tauright)leftlangle Aleft(beta-tauright)Bleft(0right)rightrangle mpThetaleft(-tauright)leftlangle Bleft(-beta-tauright)Aleft(0right)rightrangle \n=  pm C_ABleft(beta-tauright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"It follows that in the case where the two operators are the same, B=A, this expression becomes C_AAleft(tauright)=pm C_AAleft(beta-tauright). Thus, a symmetrized/antisymmetrized kernel is needed.","category":"page"},{"location":"derivations/#Spectral-Functions","page":"Derivations","title":"Spectral Functions","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"We define our spectral function as","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nrho_ABleft(omegaright)=  frac2piZleft(1mp e^-betaomegaright)sum_nme^-beta E_nleftlangle mlefthatArightnrightrangle leftlangle nlefthatBrightmrightrangle deltaleft(omega+E_m-E_nright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"We will swap indices and then take omegarightarrow -omega","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nrho_ABleft(omegaright)=  frac2piZleft(1mp e^-betaomegaright)sum_nme^-beta E_nleftlangle mlefthatArightnrightrangle leftlangle nlefthatBrightmrightrangle deltaleft(omega+E_m-E_nright)\n=  -frac2piZleft(1mp e^-betaomegaright)sum_nme^-beta E_mleftlangle nlefthatBrightmrightrangle leftlangle mlefthatArightnrightrangle deltaleft(omega+E_n-E_mright)\n=  -rho_BAleft(-omegaright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"For both fermions and bosons we pick up a minus sign for two different reasons: bosons because of the inverted Bose factor 1-e^-betaomegarightarrow e^-betaomega-1 and the fermion case's sign flips from the operator order flip. ","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"In the case of A=B we get rho(omega)=-rho(omega)","category":"page"},{"location":"derivations/#Kernels-for-\\tau\\rightarrow\\omega","page":"Derivations","title":"Kernels for taurightarrowomega","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"Spectral functions are defined in such a way as to reflect what is measured in experiment. For example, let's look at the correlation function which is also the electron Green's function","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nG_alphabetaleft(tauright)=  -leftlangle T_taua_alphaleft(tauright)a_beta^daggerleft(0right)rightrangle \nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"This relates back to the spectral function for the electron occupation energies through the relation","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nGleft(mathbfksigmatau0right)=  -leftlangle c_mathbfksigmaleft(tauright)c_mathbfksigma^daggerleft(0right)rightrangle \n=  -frac1Zsum_nleftlangle nlefte^-beta He^tau Hc_mathbfksigmae^-tau Hc_mathbfksigma^daggerrightnrightrangle \n=  -frac1Zsum_ne^-beta E_ne^tau E_nleftlangle nleftc_mathbfksigmae^-tau Hc_mathbfksigma^daggerrightnrightrangle \n=  -frac1Zsum_mne^-beta E_ne^tauleft(E_n-E_mright)leftlangle nleftc_mathbfksigmarightmrightrangle leftlangle mleftc_mathbfksigma^daggerrightnrightrangle \n=  -frac1Zsum_mne^-beta E_ne^tauleft(E_n-E_mright)leftleftlangle mleftc_mathbfksigma^daggerrightnrightrangle right^2\n=  -int_-infty^inftyfracdomega2pifrace^-omegatau1+e^-betaomegaIm Gleft(mathbfksigmaomegaright)\n=  int_-infty^inftyfracdomega2frace^-omegatau1+e^-betaomegaAleft(mathbfksigmaomegaright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"This gives us the fermionic kernel of frac12frace^-omegatau1+e^-betaomega. Doing the same treatment for antisymmetric fermionic correlations, bosonic correlations, and symmetric bosonic correlations we get the following kernels:","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nK_fermionleft(omegatauright)=  frac12frace^-tauomega1+e^-betaomega\nK_fermionantisymleft(omegatauright)=  frac12frace^-tauomega-e^-left(beta-tauright)omega1+e^-betaomega\nK_bosonasymleft(omegatauright)=  frac12frace^-tauomega1-e^-betaomega\nK_bosonsymleft(omegatauright)=  frac12frace^-tauomega+e^-left(beta-tauright)1-e^-betaomega\nendalign*","category":"page"},{"location":"derivations/#Kernel-modifications-for-AC-methods","page":"Derivations","title":"Kernel modifications for AC methods","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"For kernel functions odd about omega=0 most AC codes will pull a factor of omega into the kernel. This makes the kernel analytic on the entire omega line but AC will give you fracrho(omega)omega. Some codes will return this value. DEAC.jl handles this in the back end and reports only rho(omega).","category":"page"},{"location":"derivations/#Normalization","page":"Derivations","title":"Normalization","text":"","category":"section"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"For the fermionic non-symmetric case it is simple to tell the zeroth moment of the spectral function","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"beginalign*\nC_ABleft(0right)+C_ABleft(betaright)=  int domega Aleft(omegaright)frace^-0omega+e^-betaomega1+e^-betaomega\nC_ABleft(0right)+C_ABleft(betaright)=  int domega Aleft(omegaright)\nendalign*","category":"page"},{"location":"derivations/","page":"Derivations","title":"Derivations","text":"For any others there is no obvious way to derive the zeroth moment. ","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"EditURL = \"<unknown>/src/examples/fermion_greens.jl\"","category":"page"},{"location":"examples/fermion_greens/#Example-1:-Fermion-Greens-function","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"","category":"section"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Usage:","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"$ julia --threads=auto fermion_greens.jl","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"SmoQyDEAC uses multithreading for parallelizing runs. Multithreading it recommended.   –threads=auto will run a thread for each core available, or 2x for hyperthreading cores","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"In this example we will take the up-spin electron Green's function output from a Determinant Quantum Monte Carlo run generated via SmoQyDQMC using the Holstein Model.","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"We note the convention that the correlation function reported as the Green's Function has ħ=1 and there is no leading negative sign","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"G(kτ)=T_τc_k(τ)c_k^(0)","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Our relation uses the time fermionic kernel such that","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"G(kτ)=int_-^ dω K(ωτ)A(ω)=int_-^ fracdω2 frace^-τω1+e^-ωβA(ω)","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Since A(ω)=-ℑG(ω)π both negative signs that would normally be in the expression and factors of π cancel.","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"On to the example:","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"# First we import all required packages\nusing SmoQyDEAC\nusing FileIO","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"We now load the data provided in our source file.","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"loadfile = joinpath(pkgdir(SmoQyDEAC), \"docs/src/examples/fermion_greens_input.jld2\")\ninput_dictionary = load(loadfile)\n\nG = input_dictionary[\"G\"];\nG_error = input_dictionary[\"σ\"];\nτs = input_dictionary[\"τs\"]; # must be evenly spaced.\nβ = input_dictionary[\"β\"];\nnothing #hide","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Make an output folder for checkpoint file and output file","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"output_directory = \"fermion_greens_output/\";\ntry\n    mkdir(output_directory);\ncatch\nend","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Define necessary parameters for the DEAC run Typically you will want at least 1,000 for numberofbins * runsperbin For speed's sake we only do 20 in this example.","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"number_of_bins = 2;\nruns_per_bin = 1 ;\noutput_file = output_directory * \"fermion_out.jld2\";\ncheckpoint_directory = output_directory;\nnω = 401;\nωmin = -10.;\nωmax = 10.;\nωs = collect(LinRange(ωmin,ωmax,nω));\nnothing #hide","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Set optional parameters","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"base_seed = 1000000;\n# Note, the seed will incement for each run.\n# Starting a new run at 1000020 will have unique output from this run\nkeep_bin_data = true;\n# If true, each bin will have it's data written to the output dictionary\n# Set to false to save disk space","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Run DEAC Algorithm","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"output_dictionary = DEAC(G,G_error,β,τs,ωs,\"time_fermionic\",number_of_bins,runs_per_bin,output_file,\n                         checkpoint_directory,base_seed=base_seed,keep_bin_data=keep_bin_data)","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Accessing output","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"# Spectral function, 1D array size (nω)\nA = output_dictionary[\"A\"];\n# Spectral function error, 1D array size (nω)\nA_σ = output_dictionary[\"σ\"];\n# ω grid, 1D array size (nω)\nωs_out = output_dictionary[\"ωs\"];\n# zeroth moment: For fermions it is G(0) + G(β) which should = 1.0. Float64s\nzeroth_calc = output_dictionary[\"zeroth_moment\"];\nzeroth_σ = output_dictionary[\"zeroth_moment_σ\"];\nexpected_zeroth = output_dictionary[\"expected_zeroth_moment\"];\n# Number of average generations to converge, Float64\navg_generations = output_dictionary[\"avg_generations\"];\nnothing #hide","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"Binned information - not available if keep_bin_data=false","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"# Bin data, 2D array size (nω,nbins)\nbin_data = output_dictionary[\"bin_data\"];\n# Standard error for each bin, 2D array size (nω,nbins)\nbin_σ = output_dictionary[\"bin_σ\"];\n# zeroth moment, 1D array (nbins)\nbin_zeroth = output_dictionary[\"bin_zeroth_moment\"];\nnothing #hide","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"The dictionary will automatically be saved","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"# Example of loading the data from the jld2\ntest_dictionary = FileIO.load(output_file)","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"This is identical to output_dictionary","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"","category":"page"},{"location":"examples/fermion_greens/","page":"Example 1: Fermion Greens function","title":"Example 1: Fermion Greens function","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#About-the-Package","page":"Home","title":"About the Package","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SmoQyDEAC utilizes the Differential Evolution for Analytic Continuation algorithm developed by Nathan S. Nichols, Paul Sokol, and Adrian Del Maestro arXiv:2201.04155.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package takes imaginary time correlation functions from condensed matter Monte Carlo simulations and provides the associated spectral function on the real axis. ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"NOTE: This package is in the experimental phase of development and is not yet published to the Julia General registry.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Open a Julia REPL environment and run the following command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] dev https://github.com/sandimas/SmoQyDEAC.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"This command clones the SmoQyDEAC.jl repository to the hidden directory .julia/dev that exists in the same directory where Julia is installed.","category":"page"},{"location":"#Running-SmoQyDEAC","page":"Home","title":"Running SmoQyDEAC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SmoQyDEAC has a simple API interface with a single callable function DEAC.","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DEAC","category":"page"},{"location":"","page":"Home","title":"Home","text":"DEAC","category":"page"},{"location":"#Kernels","page":"Home","title":"Kernels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For now there are three supported kernels","category":"page"},{"location":"","page":"Home","title":"Home","text":"time_fermionic=frac12frace^-tauomega1+e^-betaomega\ntime_bosonic=frac12frace^-tauomega1-e^-betaomega\ntime_bosonic_symmetric=frac12frace^-tauomega+e^-(beta-tau)omega1-e^-betaomega","category":"page"},{"location":"#Multithreading","page":"Home","title":"Multithreading","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SmoQyDEAC utilizes Julia's Threads.@threads multithreading capability. To take advantage of this run your code using  $ julia --threads=auto yourscript.jl auto will automatically use any available cores or hyperthreads. You can set the value to a fixed number as you wish.","category":"page"},{"location":"#Tips,-tricks-and-caveats","page":"Home","title":"Tips, tricks and caveats","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You will likely never need to adjust any of the Optional Algorithm Arguments from the default. Many are merely initial values and will be updated stochastically early and often in the code.\nThe DEAC algorithm can have edge effects where it places spectral weight on the first or last ω point. This occurs when there is spectral weight just outside of your range of ωs. The solution is simply expanding the range of your output energies.\nFor bosonic correlations SmoQyDEAC returns the spectral function, e.g. B(omega) not fracB(omega)omega as some MaxEnt codes do.\nDifferent simulation codes may report correlation functions slightly differently. E.g. for SmoQyDQMC phonon_greens =langle X(tau)X(0)rangle not the actual phonon green's function of -2Omega_0langle X(tau)X(0)rangle. While the negative sign will cancel out by our choice of Kernel, you may need to postprocess the spectral function you recover. In this case Brightarrow dfracB2Omega_0","category":"page"}]
}
