# =============================================================================
# Amazon FPGA Hardware Development Kit
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License"). You may not use
# this file except in compliance with the License. A copy of the License is
# located at
#
#    http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
# implied. See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


# Add Small_Shell P&R constraints here
# BitTF: hodiny jadra (MMCM v CL) jsou asynchronni ke clk_main_a0; prechody resi
# cl_axi_clock_converter(_light) a dvoustupnove synchronizatory (ASYNC_REG) urovnovych signalu.
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins -hier -filter {NAME =~ */CORE_MMCM/CLKOUT0}]]
