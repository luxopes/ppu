# SDC template for PPU project
# Replace <CLOCK_PIN> and <CLOCK_PERIOD> with real signals/values

# Create clock (example)
create_clock -name clk -period <CLOCK_PERIOD> [get_ports {clk}]

# Input/output port timing
set_input_delay -clock clk -max 2 [get_ports {*}]
set_output_delay -clock clk -max 2 [get_ports {*}]

# Example false path / multicycle path
# set_false_path -from [get_ports {async_sig}] -to [get_ports {sync_sig}]
# set_multicycle_path 2 -setup -from [get_ports {bus_addr}] -to [get_ports {bus_data}]

# Operating conditions and wire load models are provided by .lib files from PDK
