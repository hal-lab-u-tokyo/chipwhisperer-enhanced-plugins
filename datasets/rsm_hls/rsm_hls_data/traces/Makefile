#
#    Copyright (C) 2025 The University of Tokyo
#    
#    File:          /datasets/rsm_hls/rsm_hls_data/traces/Makefile
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  21-06-2025 06:38:27
#    Last Modified: 21-06-2025 06:51:32
#

PART_SIZE ?= 100

SHELL := bash

ifeq ($(shell expr $(PART_SIZE) \> 100), 1)
$(error PART_SIZE=$(PART_SIZE) must be 100 or less)
endif

.PHONY: all clean distclean extracted_archive_marker

all: $(foreach i, $(shell seq 0 $(shell expr $(PART_SIZE) - 1)), part_$(i)_keylist.npy) \
	  $(foreach i, $(shell seq 0 $(shell expr $(PART_SIZE) - 1)), part_$(i)_knownkey.npy) \
	  $(foreach i, $(shell seq 0 $(shell expr $(PART_SIZE) - 1)), part_$(i)_textin.npy) \
	  $(foreach i, $(shell seq 0 $(shell expr $(PART_SIZE) - 1)), part_$(i)_textout.npy) \
	  $(foreach i, $(shell seq 0 $(shell expr $(PART_SIZE) - 1)), part_$(i)_traces.npy)

-include ./Makefile.deps

rsm_hls.tar:
	wget https://zenodo.org/records/15707560/files/rsm_hls.tar

extracted_archive_marker: rsm_hls.tar
	if [ ! -d part_0_keylist.npy.xz ]; then \
		tar -xf rsm_hls.tar; \
	else \
		echo "rsm_hls.tar already extracted"; \
	fi

part_%_keylist.npy: part_%_keylist.npy.xz
	unxz -k part_$*_keylist.npy.xz
part_%_knownkey.npy: part_%_knownkey.npy.xz
	unxz -k part_$*_knownkey.npy.xz
part_%_textin.npy: part_%_textin.npy.xz
	unxz -k part_$*_textin.npy.xz
part_%_textout.npy: part_%_textout.npy.xz
	unxz -k part_$*_textout.npy.xz
part_%_traces.npy: part_%_traces.npy.xz
	unxz -k part_$*_traces.npy.xz

distclean: clean
	$(RM) -f *.npy *.npy.xz

clean:
	$(RM) -f *.npy.xz