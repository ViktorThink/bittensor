# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import argparse
import bittensor
from tqdm import tqdm
from fuzzywuzzy import fuzz
from rich.align import Align
from rich.table import Table
from rich.prompt import Prompt
from typing import List, Optional, Dict
from .utils import get_hotkey_wallets_for_wallet, get_coldkey_wallets_for_path, get_all_wallets_for_path
console = bittensor.__console__

class OverviewCommand:


    @staticmethod
    def run( cli ):
        r""" Prints an overview for the wallet's colkey.
        """
        console = bittensor.__console__
        wallet = bittensor.wallet( config = cli.config )
        subtensor: 'bittensor.Subtensor' = bittensor.subtensor( config = cli.config )

        all_hotkeys = []
        total_balance = bittensor.Balance(0)
        
        # We are printing for every coldkey.
        if cli.config.all:
            cold_wallets = get_coldkey_wallets_for_path(cli.config.wallet.path)
            for cold_wallet in tqdm(cold_wallets, desc="Pulling balances"):
                if cold_wallet.coldkeypub_file.exists_on_device() and not cold_wallet.coldkeypub_file.is_encrypted():
                    total_balance = total_balance + subtensor.get_balance( cold_wallet.coldkeypub.ss58_address )
            all_hotkeys = get_all_wallets_for_path( cli.config.wallet.path )
        else:
            # We are only printing keys for a single coldkey
            coldkey_wallet = bittensor.wallet( config = cli.config )
            if coldkey_wallet.coldkeypub_file.exists_on_device() and not coldkey_wallet.coldkeypub_file.is_encrypted():
                total_balance = subtensor.get_balance( coldkey_wallet.coldkeypub.ss58_address )
            if not coldkey_wallet.coldkeypub_file.exists_on_device():
                console.print("[bold red]No wallets found.")
                return
            all_hotkeys = get_hotkey_wallets_for_wallet( coldkey_wallet )

        # We are printing for a select number of hotkeys from all_hotkeys.

        if cli.config.wallet.get('hotkeys', []):
            if not cli.config.get('all_hotkeys', False):
                # We are only showing hotkeys that are specified.
                all_hotkeys = [hotkey for hotkey in all_hotkeys if hotkey.hotkey_str in cli.config.wallet.hotkeys]
            else:
                # We are excluding the specified hotkeys from all_hotkeys.
                all_hotkeys = [hotkey for hotkey in all_hotkeys if hotkey.hotkey_str not in cli.config.wallet.hotkeys]

        # Check we have keys to display.
        if len(all_hotkeys) == 0:
            console.print("[red]No wallets found.[/red]")
            return

        # Pull neuron info for all keys.            
        neurons: Dict[str, List[bittensor.NeuronInfo, bittensor.Wallet]] = {}
        block = subtensor.block

        netuids = subtensor.get_all_subnet_netuids()
        for netuid in netuids:
            neurons[str(netuid)] = []

        with console.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(cli.config.subtensor.get('network', bittensor.defaults.subtensor.network))):
            for netuid in netuids:
                all_neurons = subtensor.neurons( netuid = netuid )
                # Map the hotkeys to uids
                hotkey_to_neurons = {n.hotkey: n.uid for n in all_neurons}
                for hot_wallet in tqdm(all_hotkeys):
                    uid = hotkey_to_neurons.get(hot_wallet.hotkey.ss58_address)
                    if uid is not None:
                        nn = all_neurons[uid]
                        neurons[str(netuid)].append( (nn, hot_wallet) )

                if len(neurons[str(netuid)]) == 0:
                    # Remove netuid from overview if no neurons are found.
                    netuids.remove(netuid)

        # Setup outer table.
        grid = Table.grid(pad_edge=False)

        title: str = ""
        if not cli.config.all:
            title = ( "[bold white italic]Wallet - {}:{}".format(cli.config.wallet.name, wallet.coldkeypub.ss58_address) )
        else:
            title = ( "[bold whit italic]All Wallets:" )

        # Add title
        grid.add_row(Align(title, vertical="middle", align="center"))

        # Generate rows per netuid
        for netuid in netuids:
            TABLE_DATA = []  
            total_stake = 0.0
            total_rank = 0.0
            total_trust = 0.0
            total_consensus = 0.0
            total_incentive = 0.0
            total_dividends = 0.0
            total_emission = 0   

            for nn, hotwallet in tqdm(neurons[str(netuid)]):
                nn: bittensor.NeuronInfo
                uid = nn.uid
                active = nn.active
                stake = sum([el[1] for el in nn.stake]).tao
                rank = nn.rank
                trust = nn.trust
                consensus = nn.consensus
                incentive = nn.incentive
                dividends = nn.dividends
                emission = int(nn.emission * 1000000000)
                last_update = int(block -  nn.last_update)
                row = [
                    hotwallet.name,
                    hotwallet.hotkey_str,
                    str(uid), 
                    str(active), 
                    '{:.5f}'.format(stake),
                    '{:.5f}'.format(rank), 
                    '{:.5f}'.format(trust), 
                    '{:.5f}'.format(consensus), 
                    '{:.5f}'.format(incentive),
                    '{:.5f}'.format(dividends),
                    '{}'.format(emission),
                    str(last_update),
                    bittensor.utils.networking.int_to_ip( nn.axon_info.ip) + ':' + str(nn.axon_info.port) if nn.axon_info.port != 0 else '[yellow]none[/yellow]', 
                    nn.hotkey
                ]
                total_stake += stake
                total_rank += rank
                total_trust += trust
                total_consensus += consensus
                total_incentive += incentive
                total_dividends += dividends
                total_emission += emission
                TABLE_DATA.append(row)
                
            total_neurons = len(neurons)

            # Add subnet header
            grid.add_row(f"Subnet: [bold white]{1}[/bold white]")

            table = Table(show_footer=False, width=cli.config.get('width', None), pad_edge=False, box=None)
            table.add_column("[overline white]COLDKEY",  str(total_neurons), footer_style = "overline white", style='bold white')
            table.add_column("[overline white]HOTKEY",  str(total_neurons), footer_style = "overline white", style='white')
            table.add_column("[overline white]UID",  str(total_neurons), footer_style = "overline white", style='yellow')
            table.add_column("[overline white]ACTIVE", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]STAKE(\u03C4)", '\u03C4{:.5f}'.format(total_stake), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]RANK", '{:.5f}'.format(total_rank), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]TRUST", '{:.5f}'.format(total_trust), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]CONSENSUS", '{:.5f}'.format(total_consensus), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]INCENTIVE", '{:.5f}'.format(total_incentive), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]DIVIDENDS", '{:.5f}'.format(total_dividends), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]EMISSION(\u03C1)", '\u03C1{}'.format(int(total_emission)), footer_style = "overline white", justify='right', style='green', no_wrap=True)
            table.add_column("[overline white]UPDATED", justify='right', no_wrap=True)
            table.add_column("[overline white]AXON", justify='left', style='dim blue', no_wrap=True) 
            table.add_column("[overline white]HOTKEY_SS58", style='dim blue', no_wrap=False)
            table.show_footer = True

            sort_by: Optional[str] = cli.config.get('sort_by', None)
            sort_order: Optional[str] = cli.config.get('sort_order', None)

            if sort_by is not None and sort_by != "":
                column_to_sort_by: int = 0
                highest_matching_ratio: int = 0
                sort_descending: bool = False # Default sort_order to ascending

                for index, column in zip(range(len(table.columns)), table.columns):
                    # Fuzzy match the column name. Default to the first column.
                    column_name = column.header.lower().replace('[overline white]', '')
                    match_ratio = fuzz.ratio(sort_by.lower(), column_name)
                    # Finds the best matching column
                    if  match_ratio > highest_matching_ratio:
                        highest_matching_ratio = match_ratio
                        column_to_sort_by = index
                
                if sort_order.lower() in { 'desc', 'descending', 'reverse'}:
                    # Sort descending if the sort_order matches desc, descending, or reverse
                    sort_descending = True
                
                def overview_sort_function(row):
                    data = row[column_to_sort_by]
                    # Try to convert to number if possible
                    try:
                        data = float(data)
                    except ValueError:
                        pass
                    return data

                TABLE_DATA.sort(key=overview_sort_function, reverse=sort_descending)

            for row in TABLE_DATA:
                table.add_row(*row)

            grid.add_row(table)


        console.clear()

        caption = "[italic][dim][white]Wallet balance: [green]\u03C4" + str(total_balance.tao)
        grid.add_row(Align(caption, vertical="middle", align="center"))

        # Print the entire table/grid
        console.print(grid, width=cli.config.get('width', None))

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        overview_parser = parser.add_parser(
            'overview', 
            help='''Show registered account overview.'''
        )
        overview_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        overview_parser.add_argument(
            '--all', 
            dest='all', 
            action='store_true', 
            help='''View overview for all wallets.''',
            default=False,
        )
        overview_parser.add_argument(
            '--no_cache', 
            dest='no_cache', 
            action='store_true', 
            help='''Set true to avoid using the cached overview from IPFS.''',
            default=False,
        )
        overview_parser.add_argument(
            '--width', 
            dest='width', 
            action='store',
            type=int, 
            help='''Set the output width of the overview. Defaults to automatic width from terminal.''',
            default=None,
        )
        overview_parser.add_argument(
            '--sort_by', 
            '--wallet.sort_by',
            dest='sort_by',
            required=False,
            action='store',
            default="",
            type=str,
            help='''Sort the hotkeys by the specified column title (e.g. name, uid, axon).'''
        )
        overview_parser.add_argument(
            '--sort_order',
            '--wallet.sort_order',
            dest="sort_order",
            required=False,
            action='store',
            default="ascending",
            type=str,
            help='''Sort the hotkeys in the specified ordering. (ascending/asc or descending/desc/reverse)'''
        )
        overview_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )  
        bittensor.wallet.add_args( overview_parser )
        bittensor.subtensor.add_args( overview_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt and not config.all:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)





      