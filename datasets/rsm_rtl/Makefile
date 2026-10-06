#
#    Copyright (C) 2025 The University of Tokyo
#    
#    File:          /datasets/rsm_rtl/Makefile
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  21-06-2025 06:35:27
#    Last Modified: 21-06-2025 12:20:44
#

..PHONY: rsm_rtl clean distclean

JOBS := $(shell \
  if command -v nproc > /dev/null; then \
    nproc; \
  else \
    sysctl -n hw.ncpu; \
  fi)

rsm_rtl:
	$(MAKE) -C rsm_rtl_data/traces -j $(JOBS)
ifdef PART_SIZE
ifeq ($(shell expr $(PART_SIZE) \<= 100), 1)
	@echo "Disabling traces for parts larger than PART_SIZE=$(PART_SIZE)"
	$(foreach i, $(shell seq $(PART_SIZE) 99), \
		sed -i -e "s/^tracefile$(i) =/#tracefile$(i) =/g" \
				-e "s/^enabled${i} =/#enabled${i} =/g" rsm_rtl.cwp; \
	)
endif
endif


clean:
	make -C rsm_rtl_data/traces clean

distclean:
	make -C rsm_rtl_data/traces distclean
	git checkout HEAD rsm_rtl.cwp