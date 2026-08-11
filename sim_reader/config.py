import os
import yaml
            
class ConfigReader:
    
    def __init__(self, config_file, mode='r'):
        """ 
        Construtor para ler/escrever arquivos de configuração YAML.

        Args:
            config_file: Caminho para o arquivo .yaml
            mode: 'r' para leitura, 'w' para escrita
        """
        self.config_file = config_file
        
        if mode == 'r':
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        elif mode == 'w':
            self.config = {}
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

    def get_all_sections(self): 
        """
        Retorna todas as seções do arquivo de configuração.
        
        Returns:
            dict: Dicionário com todas as seções e seus valores.
        """
        return self.config

    def get_section(self, section, options=None, warnings=True):
        """ 
        Retorna as opções de uma seção específica.

        Args:
            section: Nome da seção
            options: Lista de opções específicas (se None, retorna todas)
            warnings: Mostrar avisos se opção não for encontrada
        
        Return:
            dict: Opções da seção.
        """
        if section not in self.config:
            raise KeyError(f"Section '{section}' not found in {self.config_file}!")

        if options is None:
            return self.config[section]
        else:
            section_options = {}
            for op in options:
                if op in self.config[section]:
                    section_options[op] = self.config[section][op]
                elif warnings:
                    print(f'Warning: {op} option not found in config file!')
            return section_options

    def write_section(self, name, section): 
        """
        Escreve uma seção no arquivo de configuração.

        Args:
            name: Nome da seção
            section: Dicionário Python contendo os dados da seção
        """
        self.config[name] = section
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    config_file = os.path.join(current_dir, 'sim_config_example.yaml')
    config = ConfigReader(config_file)
    print(config.get_all_sections())