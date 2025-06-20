#
#    Copyright (C) 2025 The University of Tokyo
#    
#    File:          /datasets/rv_mask/Makefile
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  21-06-2025 02:05:56
#    Last Modified: 21-06-2025 03:15:23
#


..PHONY: rv_mask rv_mask_small clean distclean

rv_mask:
	$(MAKE) -C rv_mask_data/traces -j `nproc`
ifdef PART_SIZE
ifeq ($(shell expr $(PART_SIZE) \<= 100), 1)
	@echo "Disabling traces for parts larger than PART_SIZE=$(PART_SIZE)"
	$(foreach i, $(shell seq $(PART_SIZE) 99), \
		sed -i -e "s/^tracefile$(i) =/#tracefile$(i) =/g" \
				-e "s/^enabled${i} =/#enabled${i} =/g" rv_mask.cwp; \
	)
endif
endif

rv_mask_small:
	$(MAKE) -C rv_mask_small_data/traces -j `nproc`
ifdef PART_SIZE
ifeq ($(shell expr $(PART_SIZE) \<= 15), 1)
	@echo "Disabling traces for parts larger than PART_SIZE=$(PART_SIZE)"
	$(foreach i, $(shell seq $(PART_SIZE) 14), \
		sed -i -e "s/^tracefile$(i) =/#tracefile$(i) =/g" \
				-e "s/^enabled${i} =/#enabled${i} =/g" rv_mask_small.cwp; \
	)
endif
endif

clean:
	make -C rv_mask_data/traces clean
	make -C rv_mask_data_small/traces clean

distclean:
	make -C rv_mask_data/traces distclean
	make -C rv_mask_small_data/traces distclean
	git reset --hard HEAD rv_mask.cwp rv_mask_small.cwp