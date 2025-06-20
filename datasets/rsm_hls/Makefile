#
#    Copyright (C) 2025 The University of Tokyo
#    
#    File:          /datasets/rsm_hls/Makefile
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  21-06-2025 06:35:27
#    Last Modified: 21-06-2025 06:50:42
#

..PHONY: rsm_hls clean distclean

rsm_hls:
	$(MAKE) -C rsm_hls_data/traces -j `nproc`
ifdef PART_SIZE
ifeq ($(shell expr $(PART_SIZE) \<= 100), 1)
	@echo "Disabling traces for parts larger than PART_SIZE=$(PART_SIZE)"
	$(foreach i, $(shell seq $(PART_SIZE) 99), \
		sed -i -e "s/^tracefile$(i) =/#tracefile$(i) =/g" \
				-e "s/^enabled${i} =/#enabled${i} =/g" rsm_hls.cwp; \
	)
endif
endif


clean:
	make -C rsm_hls_data/traces clean

distclean:
	make -C rsm_hls_data/traces distclean
	git reset --hard HEAD rsm_hls.cwp